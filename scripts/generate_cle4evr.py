#!/usr/bin/env python3
r"""Render the CLE4EVR scenes (two objects with the same shape and color).

Runs inside Blender::

    xvfb-run -a blender -noaudio -b -P scripts/generate_cle4evr.py -- \
        --output-dir fonesy/data/cle4evr \
        --n-train 5000 --n-val 2000 --n-test 5000 \
        --properties_json scripts/cle4evr_config/properties.json \
        --render_num_samples 128 --use_gpu 1 --seed 0

Every sample satisfies the ``cle4evr`` oracle by construction. The scene
rendering helpers (``render_scene``, ``add_random_objects``, ...) are taken
from rsbench-code's ``rssgen/clevr/clevr_renderer.py`` (BSD license,
Copyright 2017-present Facebook, Inc.), which cannot be imported inside
Blender's Python; ``clevr_utils.py`` is imported from the rsbench-code clone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLEVR_DIR = _REPO_ROOT / "external" / "rsbench-code" / "rssgen" / "rssgen" / "clevr"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_CLEVR_DIR))

import bpy
import bpy_extras
from mathutils import Vector

import clevr_utils
from fonesy.utils.oracles import load_oracle

oracle = load_oracle("cle4evr")


def log(level: str, *parts) -> None:
    print(f"[{level}]", *parts, flush=True)


# ─── Vendored from clevr_renderer.py (see module docstring) ────────────────


def close_unused_files():
    all_data_blocks = (
        set(bpy.data.objects)
        | set(bpy.data.meshes)
        | set(bpy.data.materials)
        | set(bpy.data.images)
    )
    used_data_blocks = set()
    for obj in bpy.context.scene.objects:
        used_data_blocks.add(obj.data)
        for slot in obj.material_slots:
            if slot.material:
                used_data_blocks.add(slot.material)
    unused_data_blocks = all_data_blocks - used_data_blocks
    for data_block in unused_data_blocks:
        bpy.data.objects.remove(data_block, do_unlink=True)
        bpy.data.meshes.remove(data_block, do_unlink=True)
        bpy.data.materials.remove(data_block, do_unlink=True)
        bpy.data.images.remove(data_block, do_unlink=True)


class Box:
    dim_x = 1
    dim_y = 1

    def __init__(self, min_x, min_y, max_x, max_y, dim_x=dim_x, dim_y=dim_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.dim_x = dim_x
        self.dim_y = dim_y

    @property
    def x(self):
        return round(self.min_x * self.dim_x)

    @property
    def y(self):
        return round(self.dim_y - self.max_y * self.dim_y)

    @property
    def width(self):
        return round((self.max_x - self.min_x) * self.dim_x)

    @property
    def height(self):
        return round((self.max_y - self.min_y) * self.dim_y)

    def to_tuple(self):
        if self.width == 0 or self.height == 0:
            return (0, 0, 0, 0)
        return (self.x, self.y, self.width, self.height)


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    mat = cam_ob.matrix_world.normalized().inverted()
    me = me_ob.to_mesh(preserve_all_data_layers=True)
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != "ORTHO"

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    me_ob.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    return Box(min_x, min_y, max_x, max_y, dim_x, dim_y)


def compute_all_relationships(scene_struct, eps=0.2):
    all_relationships = {}
    for name, direction_vec in scene_struct["directions"].items():
        if name == "above" or name == "below":
            continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct["objects"]):
            coords1 = obj1["3d_coords"]
            related = set()
            for j, obj2 in enumerate(scene_struct["objects"]):
                if obj1 == obj2:
                    continue
                coords2 = obj2["3d_coords"]
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


COLOR_IDX, SHAPE_IDX, MATERIAL_IDX, SIZE_IDX = 0, 1, 2, 3


def add_random_objects(scene_struct, num_objects, args, camera, world_to_generate):
    """Same as upstream, minus the (unused here) shape_color_combos_json and
    no-occlusion re-placement paths — we always pass world_to_generate and
    never set --no-occlusion."""
    with open(args.properties_json, "r") as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties["colors"].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        size_kv_mapping = {k: v for k, v in properties["sizes"].items()}
        object_kv_mapping = {k: v for k, v in properties["shapes"].items()}
        material_kv_mapping = {k: v for k, v in properties["materials"].items()}

    positions = []
    objects = []
    blender_objects = []

    for i in range(num_objects):
        size_name = world_to_generate[i][SIZE_IDX]
        r = size_kv_mapping[size_name]

        num_tries = 0
        while True:
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    clevr_utils.delete_object(obj)
                log("warning", "failed to place the object!")
                return add_random_objects(
                    scene_struct, num_objects, args, camera, world_to_generate
                )

            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            dists_good = True
            margins_good = True
            for xx, yy, rr in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < args.min_dist:
                    dists_good = False
                    break
                for direction_name in ["left", "right", "front", "behind"]:
                    direction_vec = scene_struct["directions"][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < args.margin:
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break

        color_name = world_to_generate[i][COLOR_IDX]
        rgba = color_name_to_rgba[color_name]
        obj_name_out = world_to_generate[i][SHAPE_IDX]
        obj_name = object_kv_mapping[obj_name_out]

        if obj_name == "Cube":
            r /= math.sqrt(2)

        theta = 360.0 * random.random()

        clevr_utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        mat_name_out = world_to_generate[i][MATERIAL_IDX]
        mat_name = material_kv_mapping[mat_name_out]
        clevr_utils.add_material(mat_name, Color=rgba)

        pixel_coords = clevr_utils.get_camera_coords(camera, obj.location)

        scene = bpy.context.scene
        cam_ob = scene.camera
        me_ob = bpy.context.object
        bound_box = camera_view_bounds_2d(bpy.context.scene, cam_ob, me_ob)

        objects.append(
            {
                "shape": obj_name_out,
                "size": size_name,
                "material": mat_name_out,
                "3d_coords": tuple(obj.location),
                "rotation": theta,
                "pixel_coords": pixel_coords,
                "color": color_name,
                "x": bound_box.x,
                "y": bound_box.y,
                "width": bound_box.width,
                "height": bound_box.height,
            }
        )

    return objects, blender_objects


def render_scene(
    args,
    num_objects,
    output_index,
    output_split,
    output_image,
    output_scene,
    world_to_generate,
):
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
    clevr_utils.load_materials(args.material_dir)

    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    if args.use_gpu == 1:
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
        cycles_prefs.compute_device_type = "CUDA"
        # Every device must be enabled explicitly, or Cycles falls back to CPU.
        cycles_prefs.get_devices()
        for device in cycles_prefs.devices:
            device.use = device.type == "CUDA"

    bpy.data.worlds["World"].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.tile_size = args.render_tile_size
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = "GPU"

    scene_struct = {
        "split": output_split,
        "image_index": output_index,
        "image_filename": os.path.basename(output_image),
        "objects": [],
        "directions": {},
    }

    bpy.ops.mesh.primitive_plane_add()
    plane = bpy.context.object

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects["Camera"].location[i] += rand(args.camera_jitter)

    camera = bpy.data.objects["Camera"]
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    clevr_utils.delete_object(plane)

    scene_struct["directions"]["behind"] = tuple(plane_behind)
    scene_struct["directions"]["front"] = tuple(-plane_behind)
    scene_struct["directions"]["left"] = tuple(plane_left)
    scene_struct["directions"]["right"] = tuple(-plane_left)
    scene_struct["directions"]["above"] = tuple(plane_up)
    scene_struct["directions"]["below"] = tuple(-plane_up)

    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Key"].location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Back"].location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Fill"].location[i] += rand(args.fill_light_jitter)

    objects, blender_objects = add_random_objects(
        scene_struct, num_objects, args, camera, world_to_generate
    )

    scene_struct["objects"] = objects
    scene_struct["relationships"] = compute_all_relationships(scene_struct)

    try:
        with clevr_utils.stdout_redirected():
            bpy.ops.render.render(write_still=True)
    except Exception as e:
        log("warning", str(e))
        close_unused_files()

    with open(output_scene, "w") as f:
        json.dump(scene_struct, f, indent=2)


# ─── CLE4EVR-specific driver (not from upstream) ────────────────────────────

_ORACLE_SHAPE_IDX = {"cube": 0, "sphere": 1, "cylinder": 2}
_ORACLE_COLOR_IDX = {"red": 0, "blue": 1, "green": 2}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_scene_blendfile", default=str(_CLEVR_DIR.parent.parent / "clevr_config" / "base_scene.blend"))
    parser.add_argument("--properties_json", required=True)
    parser.add_argument("--shape_dir", default=str(_CLEVR_DIR.parent.parent / "clevr_config" / "shapes"))
    parser.add_argument("--material_dir", default=str(_CLEVR_DIR.parent.parent / "clevr_config" / "materials"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0, help="Global index offset (for sharding a split across processes)")
    parser.add_argument("--count", type=int, default=None, help="Render only this many images per split, starting at --start-idx (default: the full --n-<split> count)")
    parser.add_argument("--min_dist", default=0.25, type=float)
    parser.add_argument("--margin", default=0.4, type=float)
    parser.add_argument("--max_retries", default=50, type=int)
    parser.add_argument("--width", default=320, type=int)
    parser.add_argument("--height", default=240, type=int)
    parser.add_argument("--use_gpu", default=1, type=int)
    parser.add_argument("--key_light_jitter", default=1.0, type=float)
    parser.add_argument("--fill_light_jitter", default=1.0, type=float)
    parser.add_argument("--back_light_jitter", default=1.0, type=float)
    parser.add_argument("--camera_jitter", default=0.5, type=float)
    parser.add_argument("--render_num_samples", default=128, type=int)
    parser.add_argument("--render_min_bounces", default=8, type=int)
    parser.add_argument("--render_max_bounces", default=8, type=int)
    parser.add_argument("--render_tile_size", default=256, type=int)
    return parser


def generate_split(args, split: str, n_split: int, oracle_shapes, oracle_colors, materials, sizes, rng: random.Random) -> dict:
    split_dir = Path(args.output_dir) / split
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    start = args.start_idx
    count = args.count if args.count is not None else n_split
    labels: dict[str, list[int]] = {}

    for k in range(start, start + count):
        shape = rng.choice(oracle_shapes)
        color = rng.choice(oracle_colors)
        mat_a, mat_b = rng.choice(materials), rng.choice(materials)
        size_a, size_b = rng.choice(sizes), rng.choice(sizes)
        world_to_generate = [
            (color, shape, mat_a, size_a),
            (color, shape, mat_b, size_b),
        ]

        filename = f"{k:05d}.png"
        image_path = images_dir / filename
        scene_path = images_dir / f"{k:05d}.scene.json"

        render_scene(
            args,
            num_objects=2,
            output_index=k,
            output_split=split,
            output_image=str(image_path),
            output_scene=str(scene_path),
            world_to_generate=world_to_generate,
        )
        scene_path.unlink(missing_ok=True)

        shape_idx = _ORACLE_SHAPE_IDX[shape]
        color_idx = _ORACLE_COLOR_IDX[color]
        assignment = [0] * 12
        assignment[shape_idx] = 1
        assignment[3 + color_idx] = 1
        assignment[6 + shape_idx] = 1
        assignment[9 + color_idx] = 1
        assert oracle.check(assignment), (
            f"Generated sample {k} in split '{split}' failed cle4evr.check() — "
            "labeling logic is broken, aborting before writing more data."
        )

        labels[filename] = [shape_idx, color_idx, shape_idx, color_idx]
        if (k - start + 1) % 100 == 0:
            log("info", f"{split}: rendered {k - start + 1}/{count}")

    return labels


def main() -> None:
    argv = clevr_utils.extract_args()
    args = build_arg_parser().parse_args(argv)

    random.seed(args.seed)

    with open(args.properties_json, "r") as f:
        properties = json.load(f)
    oracle_shapes = [s for s in properties["shapes"] if s in _ORACLE_SHAPE_IDX]
    oracle_colors = [c for c in properties["colors"] if c in _ORACLE_COLOR_IDX]
    materials = list(properties["materials"].keys())
    sizes = list(properties["sizes"].keys())
    assert set(oracle_shapes) == set(_ORACLE_SHAPE_IDX), (
        f"--properties_json must define exactly {sorted(_ORACLE_SHAPE_IDX)} shapes, got {oracle_shapes}"
    )
    assert set(oracle_colors) == set(_ORACLE_COLOR_IDX), (
        f"--properties_json must define exactly {sorted(_ORACLE_COLOR_IDX)} colors, got {oracle_colors}"
    )

    splits = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    n_total = sum(splits.values())
    log("info", f"Generating {n_total} CLE4EVR images ({args.n_train} train / {args.n_val} val / {args.n_test} test)...")

    for split, n_split in splits.items():
        rng = random.Random(args.seed + hash(split) % 10_000)
        labels = generate_split(args, split, n_split, oracle_shapes, oracle_colors, materials, sizes, rng)

        split_dir = Path(args.output_dir) / split
        labels_path = split_dir / "labels.json"
        if args.start_idx == 0 and (args.count is None or args.count == n_split):
            split_dir.mkdir(parents=True, exist_ok=True)
            labels_path.write_text(json.dumps(labels, indent=2))
        else:
            shard_path = split_dir / f"labels_shard_{args.start_idx}_{args.count}.json"
            shard_path.write_text(json.dumps(labels, indent=2))
            log("info", f"Sharded run: wrote {shard_path}, merge shards into labels.json separately.")

        log("info", f"{split}: {len(labels)} samples verified and written to {split_dir}")

    log("info", f"Done. {n_total}/{n_total} samples satisfy cle4evr.check() — all valid.")


if __name__ == "__main__":
    main()
