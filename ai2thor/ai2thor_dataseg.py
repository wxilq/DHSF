import ai2thor.controller
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
import random
import math
from collections import defaultdict
import time

base_dir = Path("")

ALL_SCENES = (
        [f"FloorPlan{i}" for i in range(1, 31)] +  # kitchen 1-30
        [f"FloorPlan{i}" for i in range(201, 231)] +  # livingroom 201-230
        [f"FloorPlan{i}" for i in range(301, 331)] +  # bedroom 301-330
        [f"FloorPlan{i}" for i in range(401, 431)]  # bathroom 401-430
)

GLOBAL_OBJECT_ID_TO_NAME = {}
GLOBAL_ID_NAME_PATH = base_dir / "id_name.json"

if GLOBAL_ID_NAME_PATH.exists():
    with open(GLOBAL_ID_NAME_PATH, 'r', encoding='utf-8') as f:
        existing_mapping = json.load(f)
        for id_str, name in existing_mapping.items():
            if name != "background":
                GLOBAL_OBJECT_ID_TO_NAME[name] = int(id_str)
else:
    print("Creating new category mapping")

print("=" * 60)
print("AI2Thor Full-Scene Dataset Generator")
print("=" * 60)
print(f"Total scenes: {len(ALL_SCENES)}")
print(f"Kitchen: FloorPlan1-30")
print(f"Living room: FloorPlan201-230")
print(f"Bedroom: FloorPlan301-330")
print(f"Bathroom: FloorPlan401-430")
print("=" * 60 + "\n")


def save_frame(event, frame_count, color_dir, label_dir):
    global GLOBAL_OBJECT_ID_TO_NAME

    rgb_frame = event.frame

    if hasattr(event, 'instance_segmentation_frame') and event.instance_segmentation_frame is not None:
        seg_frame = event.instance_segmentation_frame
        semantic_mask = np.zeros((512, 512), dtype=np.uint8)

        objects = event.metadata['objects']
        object_id_to_color = event.object_id_to_color

        for obj in objects:
            if obj['visible']:
                object_id = obj['objectId']
                object_type = obj['objectType']

                if object_type not in GLOBAL_OBJECT_ID_TO_NAME:
                    GLOBAL_OBJECT_ID_TO_NAME[object_type] = len(GLOBAL_OBJECT_ID_TO_NAME) + 1

                class_id = GLOBAL_OBJECT_ID_TO_NAME[object_type]

                if object_id in object_id_to_color:
                    color = object_id_to_color[object_id]
                    if isinstance(color, (tuple, list)):
                        color = np.array(color[:3])
                    elif isinstance(color, str):
                        color = np.array([int(x) for x in color.split('|')])

                    mask = np.all(seg_frame == color, axis=-1)
                    semantic_mask[mask] = class_id

        rgb_image = Image.fromarray(rgb_frame)
        rgb_path = color_dir / f"{frame_count:06d}.jpg"
        rgb_image.save(rgb_path, quality=95)

        seg_image = Image.fromarray(semantic_mask, mode='L')
        seg_path = label_dir / f"{frame_count:06d}.png"
        seg_image.save(seg_path)

        return True

    return False


def execute_and_save(controller, action, degrees, frame_count, color_dir, label_dir):
    if degrees is not None:
        event = controller.step(action=action, degrees=degrees)
    else:
        event = controller.step(action=action)

    saved = save_frame(event, frame_count, color_dir, label_dir)
    if saved:
        frame_count += 1

    return event, frame_count, event.metadata['lastActionSuccess']


def calculate_distance(pos1, pos2):
    return math.sqrt(
        (pos1['x'] - pos2['x']) ** 2 +
        (pos1['z'] - pos2['z']) ** 2
    )


def position_to_grid(pos, grid_size=0.8):
    return (int(pos['x'] / grid_size), int(pos['z'] / grid_size))


def create_coverage_grid(reachable_positions, grid_size=0.8):
    grid_dict = defaultdict(list)

    for pos in reachable_positions:
        grid_key = position_to_grid(pos, grid_size)
        grid_dict[grid_key].append(pos)

    grid_centers = []
    for grid_key, positions in grid_dict.items():
        center_x = np.mean([p['x'] for p in positions])
        center_z = np.mean([p['z'] for p in positions])

        center_pos = min(positions,
                         key=lambda p: (p['x'] - center_x) ** 2 + (p['z'] - center_z) ** 2)
        grid_centers.append(center_pos)

    return grid_centers


def slow_rotate(controller, total_degrees, frame_count, color_dir, label_dir, direction='RotateRight'):
    step_degrees = 2
    num_steps = abs(int(total_degrees / step_degrees))

    for i in range(num_steps):
        event, frame_count, _ = execute_and_save(
            controller, direction, step_degrees, frame_count, color_dir, label_dir
        )

    return frame_count


def slow_look_updown(controller, total_degrees, frame_count, color_dir, label_dir, direction='LookDown'):
    step_degrees = 2
    num_steps = abs(int(total_degrees / step_degrees))

    for i in range(num_steps):
        event, frame_count, _ = execute_and_save(
            controller, direction, step_degrees, frame_count, color_dir, label_dir
        )

    return frame_count


def simple_scan(controller, frame_count, color_dir, label_dir, scan_type='quick'):
    if scan_type == 'quick':
        frame_count = slow_rotate(
            controller, 90, frame_count, color_dir, label_dir, 'RotateRight'
        )

    elif scan_type == 'medium':
        frame_count = slow_rotate(
            controller, 180, frame_count, color_dir, label_dir, 'RotateRight'
        )

    else:
        frame_count = slow_rotate(
            controller, 120, frame_count, color_dir, label_dir, 'RotateRight'
        )

        frame_count = slow_look_updown(
            controller, 15, frame_count, color_dir, label_dir, 'LookDown'
        )
        frame_count = slow_rotate(
            controller, 120, frame_count, color_dir, label_dir, 'RotateRight'
        )

        frame_count = slow_look_updown(
            controller, 15, frame_count, color_dir, label_dir, 'LookUp'
        )
        frame_count = slow_rotate(
            controller, 120, frame_count, color_dir, label_dir, 'RotateRight'
        )

    return frame_count


def walk_to_position(controller, target_pos, frame_count, color_dir, label_dir, max_steps=50):
    steps = 0

    while steps < max_steps:
        current_pos = controller.last_event.metadata['agent']['position']
        distance = calculate_distance(current_pos, target_pos)

        if distance < 0.3:
            break

        event, frame_count, success = execute_and_save(
            controller, 'MoveAhead', None, frame_count, color_dir, label_dir
        )
        steps += 1

        if not success:
            turn_dir = random.choice(['RotateRight', 'RotateLeft'])
            turn_angle = random.choice([45, 60])
            frame_count = slow_rotate(
                controller, turn_angle, frame_count, color_dir, label_dir, turn_dir
            )

    return frame_count


def scan_single_scene(scene_name):
    print(f"\n{'=' * 60}")
    print(f"Scanning scene: {scene_name}")
    print(f"{'=' * 60}")

    try:
        controller = ai2thor.controller.Controller(
            scene=scene_name,
            gridSize=0.02,
            width=512,
            height=512,
            visibilityDistance=1.5,
            renderInstanceSegmentation=True,
            snapToGrid=False,
            rotateStepDegrees=2
        )
    except Exception as e:
        print(f"Failed to load scene {scene_name}: {e}")
        return

    reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]

    grid_centers = create_coverage_grid(reachable_positions, grid_size=0.8)

    sorted_centers = [grid_centers[0]]
    remaining = set(range(1, len(grid_centers)))

    while remaining:
        last_pos = sorted_centers[-1]
        nearest_idx = min(remaining,
                          key=lambda i: calculate_distance(last_pos, grid_centers[i]))
        sorted_centers.append(grid_centers[nearest_idx])
        remaining.remove(nearest_idx)

    total_frames = 0
    for grid_idx, target_pos in enumerate(sorted_centers):
        grid_dir = base_dir / f"scene_{scene_name}_{grid_idx:03d}"
        color_dir = grid_dir / "color"
        label_dir = grid_dir / "label_uint8"

        color_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        frame_count = 0

        controller.step(
            action="Teleport",
            position=target_pos,
            rotation={"x": 0, "y": random.randint(0, 3) * 90, "z": 0}
        )
        save_frame(controller.last_event, frame_count, color_dir, label_dir)
        frame_count += 1

        is_corner = grid_idx % 15 == 0
        is_center = grid_idx % 8 == 0

        if is_corner:
            scan_type = 'full'
            expected_frames = "~195"
        elif is_center:
            scan_type = 'medium'
            expected_frames = "~90"
        else:
            scan_type = 'quick'
            expected_frames = "~45"

        print(f"  Grid [{grid_idx + 1}/{len(sorted_centers)}] - {scan_type.upper()} - expected {expected_frames} frames", end=" ")

        frame_count = simple_scan(controller, frame_count, color_dir, label_dir, scan_type)

        print(f"-> {frame_count} frames saved")
        total_frames += frame_count

    controller.stop()
    print(f"\n{scene_name} done. Total frames: {total_frames}")

    return total_frames


def save_global_mapping():
    final_mapping = {"0": "background"}
    final_mapping.update({str(v): k for k, v in GLOBAL_OBJECT_ID_TO_NAME.items()})

    with open(GLOBAL_ID_NAME_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_mapping, f, indent=2, ensure_ascii=False)

    print(f"\nGlobal category mapping saved: {GLOBAL_ID_NAME_PATH}")
    print(f"Total object categories: {len(GLOBAL_OBJECT_ID_TO_NAME) + 1}")


if __name__ == "__main__":
    start_time = time.time()
    total_frames_all = 0
    completed_scenes = 0

    print("\nStarting batch scan...\n")

    for idx, scene in enumerate(ALL_SCENES, 1):
        print(f"\nProgress: [{idx}/{len(ALL_SCENES)}] ({idx / len(ALL_SCENES) * 100:.1f}%)")

        try:
            frames = scan_single_scene(scene)
            if frames:
                total_frames_all += frames
                completed_scenes += 1

                if completed_scenes % 5 == 0:
                    save_global_mapping()
                    print(f"Checkpoint saved ({completed_scenes}/{len(ALL_SCENES)})")

        except Exception as e:
            print(f"Scene {scene} failed: {e}")
            continue

    save_global_mapping()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("All scenes completed.")
    print("=" * 60)
    print(f"Successful scenes: {completed_scenes}/{len(ALL_SCENES)}")
    print(f"Total frames: {total_frames_all:,}")
    print(f"Object categories: {len(GLOBAL_OBJECT_ID_TO_NAME) + 1}")
    print(f"Elapsed time: {elapsed / 3600:.2f} hours")
    print(f"Average speed: {elapsed / completed_scenes:.1f} sec/scene")
    print("=" * 60)

    print("\nDataset location:")
    print(f"  Scene data: {base_dir}/scene_*")
    print(f"  Category mapping: {GLOBAL_ID_NAME_PATH}")

    print("\nDiscovered object categories:")
    final_mapping = {"0": "background"}
    final_mapping.update({str(v): k for k, v in GLOBAL_OBJECT_ID_TO_NAME.items()})
    for class_id, class_name in sorted(final_mapping.items(), key=lambda x: int(x[0]))[:20]:
        print(f"  {class_id}: {class_name}")
    if len(final_mapping) > 20:
        print(f"  ... and {len(final_mapping) - 20} more categories")

    print("\nDataset generation complete.")