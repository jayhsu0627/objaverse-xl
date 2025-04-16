"""Blender script to render images of 3D models."""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector

import csv
import pandas as pd
import ast
import colorsys
import bmesh
import shutil

import imageio
import time

# Set Cycles render engine
bpy.context.scene.render.engine = 'CYCLES'

# Get Cycles preferences
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.get_devices()  # Necessary to populate the device list

# Log the device info to stdout
print("=== Blender Cycles Devices ===")
print(f"Compute device type: {prefs.compute_device_type}")
for device in prefs.devices:
    print(f"- {device.name} ({device.type}) | Use: {device.use}")
print("=== End Devices ===")

# Optional: force GPU usage if available
prefs.compute_device_type = 'CUDA'  # or 'OPTIX' if supported
for device in prefs.devices:
    if device.type in ['CUDA', 'OPTIX']:
        device.use = True
bpy.context.scene.cycles.device = 'GPU'

def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = cycles_preferences.devices

    if not devices:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []
    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)
            print('activated gpu', device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus


enable_gpus("CUDA")

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    """Samples a point on a sphere with the given radius.

    Args:
        radius (float): Radius of the sphere.

    Returns:
        Tuple[float, float, float]: A point on the sphere.
    """
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def _sample_spherical(
    radius_min: float = 1.5,
    radius_max: float = 2.0,
    maxz: float = 1.6,
    minz: float = -0.75,
) -> np.ndarray:
    """Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    correct = False
    vec = np.array([0, 0, 0])
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def randomize_camera(
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 2.2,
    minz: float = -2.2,
    only_northern_hemisphere: bool = False,
) -> bpy.types.Object:
    """Randomizes the camera location and rotation inside of a spherical shell.

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """

    x, y, z = _sample_spherical(
        radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
    )
    camera = bpy.data.objects["Camera"]

    # only positive z
    if only_northern_hemisphere:
        z = abs(z)

    camera.location = Vector(np.array([x, y, z]))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


def _set_camera_at_size(i: int, scale: float = 1.5) -> bpy.types.Object:
    """Debugging function to set the camera on the 6 faces of a cube.

    Args:
        i (int): Index of the face of the cube.
        scale (float, optional): Scale of the cube. Defaults to 1.5.

    Returns:
        bpy.types.Object: The camera object.
    """
    if i == 0:
        x, y, z = scale, 0, 0
    elif i == 1:
        x, y, z = -scale, 0, 0
    elif i == 2:
        x, y, z = 0, scale, 0
    elif i == 3:
        x, y, z = 0, -scale, 0
    elif i == 4:
        x, y, z = 0, 0, scale
    elif i == 5:
        x, y, z = 0, 0, -scale
    else:
        raise ValueError(f"Invalid index: i={i}, must be int in range [0, 5].")
    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    color: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.color = color
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def randomize_lighting() -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 1),
        color = (1, 1, 1, 1),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([3, 4, 5]),
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 1),
        color = (1, 1, 1, 1),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([2, 3, 4]),
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 1),
        color = (1, 1, 1, 1),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([3, 4, 5]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 1),
        color = (1, 1, 1, 1),
        rotation=(3.14159, 0, 0),
        energy=random.choice([1, 2, 3]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def randomize_lighting_pd(light_type, color, rotation) -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type= light_type,
        location=(0, 0, 9),
        color = color,
        rotation= rotation,
        energy=random.choice([600, 800, 1000]),
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type= light_type,
        location=(0, 0, 9),
        color = color,
        rotation=rotation,
        energy=random.choice([400, 600, 800]),
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 9),
        color = color,
        rotation= rotation,
        energy=random.choice([600, 800, 1000]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type= light_type,
        location=(0, 0, 9),
        color = color,
        rotation= rotation,
        energy=random.choice([200, 400, 600]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )

def spotlight_on_object(position, rotation) -> Dict[str, bpy.types.Object]:

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    (x, y, z) = position
    color = (1, 1, 1, 1)
    
    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type= 'SPOT',
        location=(x, y, z),
        color = color,
        rotation= (0, 0, 0),
        energy=random.uniform(2200, 2800),
    )
    spot_angle = random.uniform(15, 30)
    key_light.data.spot_size = math.radians(spot_angle)
    shadow_soft_size = random.uniform(0.1, 1)
    key_light.data.shadow_soft_size = shadow_soft_size
    # key_light.data.lightgroup = "KeyGroup"  # assign a custom group name
    # bpy.context.view_layer.update()

    # Create fill light, turn off
    fill_light = _create_light(
        name="Fill_Light",
        light_type= 'POINT',
        location=(0, 0, 9),
        color = color,
        rotation= (0, 0, 0),
        energy= 0,
    )

    # Create rim light, that works for ambient light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, z),
        color = color,
        rotation=  (0, 0, 0),
        energy=random.choice([0.5, 1, 2, 3]),
    )
    rim_light.data.cycles.cast_shadow = False
#    bpy.context.view_layer.update()

    # Create bottom light, turn off
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type= 'POINT',
        location=(0, 0, 9),
        color = color,
        rotation= (0, 0, 0),
        energy= 0,
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene and joins all parts into a single object.
    
    Args:
        object_path (str): Path to the model file.
        
    Raises:
        ValueError: If the file extension is not supported.
        
    Returns:
        None
    """
    # Store initial object count to identify newly imported objects
    initial_objects = set(bpy.data.objects)
    
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")
    
    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz
        import_usdz(bpy.context, filepath=object_path, materials=True, animations=True)
    else:
        # load from existing import functions
        import_function = IMPORT_FUNCTIONS[file_extension]
        if file_extension == "blend":
            import_function(directory=object_path, link=False)
        elif file_extension in {"glb", "gltf"}:
            import_function(filepath=object_path, merge_vertices=True)
        else:
            import_function(filepath=object_path)
    
    # Identify newly imported objects
    new_objects = set(bpy.data.objects) - initial_objects
    
    if new_objects:
        # Ensure we're in object mode
        if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        
        # Find a mesh object to be the main object (preferably the first mesh)
        main_object = None
        for obj in new_objects:
            if obj.type == 'MESH':
                main_object = obj
                break
        
        # If no mesh found, use the first object
        if not main_object and new_objects:
            main_object = list(new_objects)[0]
        
        if main_object:
            # Make the main object active
            bpy.context.view_layer.objects.active = main_object
            
            # Select all new objects
            for obj in new_objects:
                obj.select_set(True)
            
            # Join all selected objects
            bpy.ops.object.join()
            
            # Rename the resulting object to match the filename
            base_name = os.path.basename(object_path).split('.')[0]
            bpy.context.active_object.name = base_name
            
            return bpy.context.active_object
    
    return None

def load_multi_objects(
    object_paths: List[str],
    join_parts: bool = True,
    area_size: float = 5.0,     # bigger so you can see them apart
) -> List[bpy.types.Object]:

    half = area_size / 2.0
    imported_roots: List[bpy.types.Object] = []

    for path in object_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        # ── remember what was in the scene ────────────────────────────────
        before = set(bpy.data.objects)

        # ── import --------------------------------------------------------
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        op  = IMPORT_FUNCTIONS[ext]
        if ext == "blend":
            op(directory=os.path.join(path, "Object"),
               filename=os.path.splitext(os.path.basename(path))[0],
               link=False)
        elif ext in {"glb", "gltf"} and bpy.app.version >= (3, 5, 0):
            op(filepath=path, merge_vertices=True)
        else:
            op(filepath=path)

        new_objs = [o for o in bpy.data.objects if o not in before]
        if not new_objs:
            continue

        # ── join ONLY the parts that belong to THIS file ──────────────────
        root = next((o for o in new_objs if o.type == "MESH"), new_objs[0])

        if join_parts and len(new_objs) > 1:
            # make sure we operate in OBJECT mode
            if bpy.context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')

            # select just the new objects
            bpy.ops.object.select_all(action='DESELECT')
            for o in new_objs:
                o.select_set(True)

            bpy.context.view_layer.objects.active = root
            bpy.ops.object.join()

        root.name = os.path.basename(path).split(".")[0]

        # ── scatter so they don't overlap visually ────────────────────────
        root.location = (
            random.uniform(-half, half),
            random.uniform(-half, half),
            0.0,
        )

        imported_roots.append(root)

        # ⚠️  DESELECT the root so it won't be re‑joined next loop
        root.select_set(False)

    bpy.context.view_layer.update()
    return imported_roots


def get_all_meshes_recursive(objects: List[bpy.types.Object]) -> List[bpy.types.Object]:
    """Recursively collects all mesh children from a list of imported objects."""
    result = []

    def recurse(obj):
        if obj.type == 'MESH':
            result.append(obj)
        for child in obj.children:
            recurse(child)

    for obj in objects:
        recurse(obj)

    return result

def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    objs = [single_obj] if single_obj else bpy.context.scene.objects
    for obj in objs:
        if obj.type != 'MESH':
            continue
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(a, b) for a, b in zip(bbox_min, coord))
            bbox_max = tuple(max(a, b) for a, b in zip(bbox_max, coord))
    if all(c == math.inf for c in bbox_min):
        raise RuntimeError("No mesh objects to compute bounding box.")
    return Vector(bbox_min), Vector(bbox_max)

def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene() -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(d - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    
    # for obj in get_scene_root_objects():
    #     obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None

def reset_scene():
    """Removes all objects except cameras and lights."""
    for obj in list(bpy.data.objects):
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)



def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs[
                                    "Base Color"
                                ].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(
    obj: bpy.types.Object, color: Tuple[float, float, float, float]
) -> None:
    """Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def apply_single_random_color_to_all_objects() -> Tuple[float, float, float, float]:
    """Applies a single random color to all objects in the scene.

    Returns:
        Tuple[float, float, float, float]: The random color that was applied to all
        objects.
    """
    rand_color = _get_random_color()
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_color_to_object(obj, rand_color)
    return rand_color


def generate_morandi_color(light=True, saturation_factor=0.3, value_factor=0.8):
    """Generate a random light or dark Morandi-style color (muted, desaturated) using HSV."""
    h = random.random()  # Random hue
    s = random.random() * saturation_factor # Reduced saturation
    v = (0.8 if light else 0.3) + random.random() * (1 - (0.8 if light else 0.3)) * value_factor # Adjusted value

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b, 1.0)  # Alpha is 1.0

def create_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.9
    return mat

def add_wall_background():
    """Add a box of walls with edges of 5 units, but only the floor at -0.5."""
    
    # Generate Morandi colors
    wall_color = generate_morandi_color(light=random.choice([True, False]))
    floor_color = generate_morandi_color(light=random.choice([True, False]))
    
    # Create materials
    wall_mat = create_material("WallMaterial", wall_color)
    floor_mat = create_material("FloorMaterial", floor_color)
    
    # Floor (at z=-0.5)
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -0.0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.data.materials.append(floor_mat)
    
    # Back wall
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, -5, 5))
    back_wall = bpy.context.active_object
    back_wall.name = "BackWall"
    back_wall.rotation_euler[0] = math.radians(90)
    back_wall.data.materials.append(wall_mat)
    
    # Side wall (left)
    bpy.ops.mesh.primitive_plane_add(size=10, location=(-5, 0, 5))
    side_wall_left = bpy.context.active_object
    side_wall_left.name = "SideWallLeft"
    side_wall_left.rotation_euler[1] = math.radians(90)
    side_wall_left.data.materials.append(wall_mat)
    
    # Side wall (right)
    bpy.ops.mesh.primitive_plane_add(size=10, location=(5, 0, 5))
    side_wall_right = bpy.context.active_object
    side_wall_right.name = "SideWallRight"
    side_wall_right.rotation_euler[1] = math.radians(90)
    side_wall_right.data.materials.append(wall_mat)
    
    # Back wall (far)
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 5, 5))
    back_wall_far = bpy.context.active_object
    back_wall_far.name = "BackWallFar"
    back_wall_far.rotation_euler[0] = math.radians(90)
    back_wall_far.data.materials.append(wall_mat)
    
    # Ceiling
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 10))
    ceiling = bpy.context.active_object
    ceiling.name = "Ceiling"
    ceiling.rotation_euler[0] = math.radians(180)
    ceiling.data.materials.append(wall_mat)


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""

    def __init__(
        self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData
    ) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()

        all_filepaths = (
            image_filepaths | material_filepaths | linked_libraries_filepaths
        )
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += (
                        len(shape_keys.key_blocks) - 1
                    )  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
        }

def load_object(object_file):
    """Load an object from a file."""
    # Get file extension
    extension = os.path.splitext(object_file)[1].lower()
    
    # Clear selection
    bpy.ops.object.select_all(action='DESELECT')
    
    # Import based on file extension
    if extension == '.glb' or extension == '.gltf':
        bpy.ops.import_scene.gltf(filepath=object_file)
    elif extension == '.fbx':
        bpy.ops.import_scene.fbx(filepath=object_file)
    elif extension == '.obj':
        bpy.ops.import_scene.obj(filepath=object_file)
    elif extension == '.stl':
        bpy.ops.import_mesh.stl(filepath=object_file)
    elif extension == '.ply':
        bpy.ops.import_mesh.ply(filepath=object_file)
    elif extension == '.usd' or extension == '.usdc' or extension == '.usda':
        bpy.ops.import_scene.usd(filepath=object_file)
    else:
        raise ValueError(f'Unsupported file extension: {extension}')
    
    # Return all selected objects (newly imported objects)
    return bpy.context.selected_objects

def recenter_object_xy(root):
    """
    Repositions the object (including all its mesh children) so that the
    overall geometric center in the X and Y directions is at (0, 0).
    This does not change the object's origin itself, but rather translates the whole
    group so that its computed center becomes centered.
    """
    bpy.context.view_layer.update()
    
    def get_all_mesh_objects(obj):
        """Recursively collects all mesh objects in the hierarchy."""
        meshes = []
        if obj.type == 'MESH':
            meshes.append(obj)
        for child in obj.children:
            meshes.extend(get_all_mesh_objects(child))
        return meshes
    
    mesh_objects = get_all_mesh_objects(root)
    if not mesh_objects:
        return  # No mesh data available to center
    
    # Collect all world-space bounding box corners from all meshes
    all_corners = []
    for mesh in mesh_objects:
        if mesh.bound_box:
            corners = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
            all_corners.extend(corners)
    
    if not all_corners:
        return

    # Compute the average X and Y values from all corners
    avg_x = sum(corner.x for corner in all_corners) / len(all_corners)
    avg_y = sum(corner.y for corner in all_corners) / len(all_corners)
    
    # Calculate the offset needed so that the geometric center becomes (0, 0)
    offset = Vector((avg_x, avg_y, 0))
    
    # Adjust the root's location so that its geometry is centered in X and Y.
    # (This moves the entire hierarchy along X and Y.)
    root.location -= Vector((offset.x, offset.y, 0))
    
    bpy.context.view_layer.update()

def load_multi_objects(
    object_paths: List[str],
    join_parts: bool = True,
    area_size: float = 8.0,
) -> List[bpy.types.Object]:
    """Loads multiple 3D objects from files, joins their mesh parts if desired, and lays them out randomly."""
    half = area_size / 2.0
    imported_roots: List[bpy.types.Object] = []
    
    model_files = object_paths
    for path in object_paths:
        imported_objects = load_object(path)
        min_distance = 2  # Minimum distance between objects (to prevent overlap)
        placed_positions = []
        
        # Restructure hierarchy to ensure proper root nodes
        root_objects = restructure_hierarchy(imported_objects)
        # Process each root object
        for root in root_objects:
            imported_roots.append(root)
            recenter_object_xy(root)
            
            # Normalize the root object (and its children)
            normalize_object(root)
            set_object_on_ground(root, ground_z=0)
            

            # Find a valid random position
            max_attempts = 50
            attempts = 0
            valid_position_found = False
            # Store original Z position
            original_z = root.location.z
            while attempts < max_attempts and not valid_position_found:
                # Generate random position within 5x5 area (only X and Y)
                x = random.uniform(-half, half)
                y = random.uniform(-half, half)
                if is_valid_position(x, y, placed_positions, min_distance):
                    root.location.x = x
                    root.location.y = y
                    placed_positions.append((x, y))
                    valid_position_found = True
                attempts += 1
            
            if not valid_position_found:
                print(f"Could not find valid position for {root.name} after {max_attempts} attempts")
                
                # Place it anyway at a random position as fallback (only X and Y)
                x = random.uniform(-area_size/2, area_size/2)
                y = random.uniform(-area_size/2, area_size/2)
                root.location = (x, y, original_z)
    
    print(f"Imported, restructured, normalized, and placed objects from {len(model_files)} files on the plane")

    return imported_roots

def restructure_hierarchy(objects):
    """Restructure the hierarchy to ensure all components have a single root node."""
    if not objects:
        return []
    
    # Identify unique top-level parents
    top_parents = {}
    for obj in objects:
        # Find the top-most parent that is part of the imported objects
        current = obj
        top_parent = current
        
        while current.parent and current.parent in objects:
            current = current.parent
            top_parent = current
            
        if top_parent not in top_parents:
            top_parents[top_parent] = []
        
        # Only add the object if it's not already a child of another object in our selection
        if not any(obj.parent == potential_parent for potential_parent in objects if potential_parent != top_parent):
            top_parents[top_parent].append(obj)
    
    # Create root nodes for each model
    root_objects = []
    
    for top_parent, children in top_parents.items():
        # For GLB files with Sketchfab structure, create a single root
        if top_parent.name.startswith("Sketchfab_model"):
            # Create an empty as the new root
            bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
            root = bpy.context.active_object
            root.name = f"Root_{top_parent.name}"
            
            # Find all 3rd level objects in the hierarchy
            third_level_objects = []
            
            for child in top_parent.children:
                # This would be the 2nd level (like carpart.obj.cleaner.materialmerger.gles)
                for grandchild in child.children:
                    # This would be the 3rd level (like Object_2, Object_3)
                    third_level_objects.append(grandchild)
            
            # If we found 3rd level objects, parent them directly to our new root
            if third_level_objects:
                for obj in third_level_objects:
                    # Store original transform
                    orig_matrix_world = obj.matrix_world.copy()
                    
                    # Set parent
                    obj.parent = root
                    
                    # Restore original world transform
                    obj.matrix_world = orig_matrix_world
                
                # Delete the original top parent and its immediate children
                # since we've reparented the 3rd level objects
                bpy.data.objects.remove(top_parent, do_unlink=True)
                
                root_objects.append(root)
            else:
                # If no 3rd level objects found, use original top parent
                root_objects.append(top_parent)
        else:
            # For other file types, use the original top parent
            root_objects.append(top_parent)
    
    return root_objects

def normalize_object(obj):
    """Normalize a single object to fit in a unit box."""
    if obj.name == "PlacementPlane":
        return
    
    # Calculate the combined bounding box of the object and all its children
    min_coords = [float('inf'), float('inf'), float('inf')]
    max_coords = [float('-inf'), float('-inf'), float('-inf')]
    
    # For the object itself
    if obj.type == 'MESH':
        for v in obj.bound_box:
            world_v = obj.matrix_world @ Vector(v)
            for i in range(3):
                min_coords[i] = min(min_coords[i], world_v[i])
                max_coords[i] = max(max_coords[i], world_v[i])
    
    # For all children
    for child in obj.children_recursive:
        if child.type == 'MESH':
            for v in child.bound_box:
                world_v = child.matrix_world @ Vector(v)
                for i in range(3):
                    min_coords[i] = min(min_coords[i], world_v[i])
                    max_coords[i] = max(max_coords[i], world_v[i])
    
    # Calculate dimensions
    dimensions = [max_coords[i] - min_coords[i] for i in range(3)]
    max_dimension = max(dimensions)
    
    if max_dimension == 0:
        return
    
    # Calculate scale factor to normalize to unit size
    scale_factor = 1.0 / max_dimension
    
    # Apply scale to the root object
    obj.scale = (obj.scale.x * scale_factor, 
                 obj.scale.y * scale_factor, 
                 obj.scale.z * scale_factor)

def set_pass_index_on_meshes(obj, pass_index_val=1):
    """
    Recursively sets the pass_index property of the object and all its children
    (only if they are of type 'MESH') to the given value.
    """
    if obj.type == 'MESH':
        print('found mesh =>>>', obj)
        obj.pass_index = pass_index_val
    for child in obj.children:
        print('here =>>>', child)
        set_pass_index_on_meshes(child, pass_index_val)

def is_valid_position(x, z, positions, min_distance):
    """Check if a position is valid (not too close to other objects)."""
    for pos in positions:
        distance = math.sqrt((x - pos[0])**2 + (z - pos[1])**2)
        if distance < min_distance:
            return False
    return True

def set_object_on_ground(root, ground_z=0):
    """
    Moves the entire object hierarchy vertically so that the overall lowest 
    point (from all mesh children) is at ground_z. This does not change 
    the origin of the root, only its world location.
    """
    # Ensure scene info is updated
    bpy.context.view_layer.update()
    
    def get_all_mesh_objects(obj):
        """
        Recursively gathers all objects of type 'MESH' in the hierarchy.
        """
        meshes = []
        if obj.type == 'MESH':
            meshes.append(obj)
        for child in obj.children:
            meshes.extend(get_all_mesh_objects(child))
        return meshes
    
    # Get all mesh objects in the hierarchy starting from the root.
    mesh_objects = get_all_mesh_objects(root)
    
    if not mesh_objects:
        print(f"No mesh objects found for {root.name}.")
        return
    
    # Collect world-space bounding box corners from all meshes.
    all_corners = []
    for mesh in mesh_objects:
        # Each mesh has its own local bounding box corners.
        if mesh.bound_box:
            corners = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
            all_corners.extend(corners)
    
    # If no corners were found, exit early.
    if not all_corners:
        print(f"No bounding box data found for {root.name}.")
        return

    # Determine the overall minimum Z value from the collected corners.
    min_z = min(corner.z for corner in all_corners)
    
    # Calculate the offset required so the lowest point is at ground_z.
    offset_z = ground_z - min_z
    print(f"Current {root.name} Z location:", root.location.z)
    print("Calculated offset_z:", offset_z)
    
    # Apply the offset to the root object's Z location.
    root.location.z += offset_z +0.001
    print(f"New {root.name} Z location:", root.location.z)

    # Update the scene again if needed.
    bpy.context.view_layer.update()

def render_object(
    object_file: str = None,
    objects_list: Optional[list]= None,
    num_renders: int = 16,
    only_northern_hemisphere: bool = False,
    output_dir: str = '~/.render',
) -> None:
    """Saves rendered images with its camera matrix and metadata of the object.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        only_northern_hemisphere (bool): Whether to only render sides of the object that
            are in the northern hemisphere. This is useful for rendering objects that
            are photogrammetrically scanned, as the bottom of the object often has
            holes.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved.

    Returns:
        None
    """
    timings = []  # to store timing for each i

    os.makedirs(output_dir, exist_ok=True)
    print(objects_list)
    print(object_file)
 
    # load the object
    if objects_list:
        print("activated")
        reset_scene()
        # roots = load_multi_objects(objects_list,
        #                    join_parts=True,
        #                    area_size=5.0)   # <- was 1.0
        
        roots = load_multi_objects(objects_list, join_parts=True)   # <- was 1.0

    else:

        if object_file.endswith(".blend"):
            bpy.ops.object.mode_set(mode="OBJECT")
            reset_cameras()
            delete_invisible_objects()
        else:
            reset_scene()
            load_object(object_file)

    # ------------------------------------------------------------------
    #  DESIGNATE THE FOCUS OBJECT
    
    # focus_obj = roots[0]              # or random.choice(roots) / by name / CLI flag
    # focus_obj.pass_index = 1          # mask index that we’ll use later

    focus_obj = roots[int(random.uniform(0, len(roots)-1))]              # or random.choice(roots) / by name / CLI flag
    set_pass_index_on_meshes(focus_obj, 1)

    # give every *other* object a different index (or 0 to ignore)
    for obj in roots:
        if obj is not focus_obj:
            obj.pass_index = 0

    context = bpy.context
    obj = bpy.context.selected_objects[0]


    spotlight_x = focus_obj.location.x
    spotlight_y = focus_obj.location.y

    context.view_layer.objects.active = obj
    context.view_layer.use_pass_z = True  # This enables the Z/Depth pass
    # obj.pass_index = 1  # Assign your desired pass index

    # Set up cameras
    cam = scene.objects["Camera"]
    cam.data.lens = 35
    cam.data.sensor_width = 32

    # Get the compositing node tree
    if bpy.context.scene.node_tree:
        # Clear all nodes
        bpy.context.scene.node_tree.nodes.clear()

    # Set up camera constraints
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

#     # # Extract the metadata. This must be done before normalizing the scene to get
#     # # accurate bounding box information.
#     # metadata_extractor = MetadataExtractor(
#     #     object_path=object_file, scene=scene, bdata=bpy.data
#     # )
#     # metadata = metadata_extractor.get_metadata()

#     # # delete all objects that are not meshes
#     # if object_file.lower().endswith(".usdz"):
#     #     # don't delete missing textures on usdz files, lots of them are embedded
#     #     missing_textures = None
#     # else:
#     #     missing_textures = delete_missing_textures()
#     # metadata["missing_textures"] = missing_textures

#     # # possibly apply a random color to all objects
#     # if object_file.endswith(".stl") or object_file.endswith(".ply"):
#     #     assert len(bpy.context.selected_objects) == 1
#     #     rand_color = apply_single_random_color_to_all_objects()
#     #     metadata["random_color"] = rand_color
#     # else:
#     #     metadata["random_color"] = None

#     # # save metadata
#     # metadata_path = os.path.join(output_dir, "metadata.json")
#     # os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
#     # with open(metadata_path, "w", encoding="utf-8") as f:
#     #     json.dump(metadata, f, sort_keys=True, indent=2)

#     # # ─────────────────────────────────────────────────────────────────────────────
#     # # objects_list already contains the absolute paths you just imported
#     # # scene        is the active Blender scene
#     # # output_dir   is where you’ll drop the JSON
#     # # ─────────────────────────────────────────────────────────────────────────────
#     # scene_metadata: list[dict] = []

#     # for object_file in objects_list:
#     #     # 1️⃣  Collect geometry‑dependent facts *before* any scaling / layout
#     #     extractor = MetadataExtractor(
#     #         object_path=object_file,
#     #         scene=scene,
#     #         bdata=bpy.data,
#     #     )
#     #     meta = extractor.get_metadata()                     # dict

#     #     # 2️⃣  Remove references to textures that aren’t on disk
#     #     if object_file.lower().endswith(".usdz"):
#     #         # textures are embedded in the USDZ container
#     #         missing = None
#     #     else:
#     #         missing = delete_missing_textures()             # int | list | None
#     #     meta["missing_textures"] = missing

#     #     # # 3️⃣  Optionally splash a random colour on STL / PLY meshes
#     #     # if object_file.lower().endswith((".stl", ".ply")):
#     #     #     # assume exactly one mesh was selected for this import
#     #     #     rand_colour = apply_single_random_color_to_all_objects()
#     #     #     meta["random_color"] = rand_colour              # e.g. [r, g, b]
#     #     # else:
#     #     #     meta["random_color"] = None

#     #     # 4️⃣  Keep track of which source file this entry refers to
#     #     meta["source_file"] = os.path.basename(object_file)

#     #     scene_metadata.append(meta)

#     # # ─────────────────────────────────────────────────────────────────────────────
#     # # Write *all* per‑object metadata into one JSON array
#     # metadata_path = os.path.join(output_dir, "metadata.json")
#     # os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

#     # with open(metadata_path, "w", encoding="utf-8") as f:
#     #     json.dump(scene_metadata, f, indent=2, sort_keys=True)

#     nodes = bpy.context.scene.node_tree.nodes
#     links = bpy.context.scene.node_tree.links

#     # Create input render layer node
#     render_layers = nodes.new('CompositorNodeRLayers')

#     # Create depth output nodes
#     depth_file_output = nodes.new(type="CompositorNodeOutputFile")
#     depth_file_output.label = 'Depth Output'
#     depth_file_output.base_path = output_dir
#     depth_file_output.file_slots[0].use_node_format = True
#     depth_file_output.format.file_format = "PNG"
#     depth_file_output.format.color_depth ='8'
#     depth_file_output.format.color_mode = "BW"

#     # Remap as other types can not represent the full range of depth.
#     nomalize = nodes.new(type="CompositorNodeNormalize")
# #    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
# #    map.offset = [-0.7]
# #    map.size = [1.4]
# #    map.use_min = True
# #    map.min = [0]

#     links.new(render_layers.outputs['Depth'], nomalize.inputs[0])
#     links.new(nomalize.outputs[0], depth_file_output.inputs[0])
# #    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

#     # Create normal output nodes
#     scale_node = nodes.new(type="CompositorNodeMixRGB")
#     scale_node.blend_type = 'MULTIPLY'
#     # scale_node.use_alpha = True
#     scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
#     links.new(render_layers.outputs['Normal'], scale_node.inputs[1])

#     bias_node = nodes.new(type="CompositorNodeMixRGB")
#     bias_node.blend_type = 'ADD'
#     # bias_node.use_alpha = True
#     bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
#     links.new(scale_node.outputs[0], bias_node.inputs[1])

#     normal_file_output = nodes.new(type="CompositorNodeOutputFile")
#     normal_file_output.label = 'Normal Output'
#     normal_file_output.base_path = output_dir
#     normal_file_output.file_slots[0].use_node_format = True
#     normal_file_output.format.file_format = "PNG"
#     links.new(bias_node.outputs[0], normal_file_output.inputs[0])

#     # Create albedo output nodes
#     alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
#     links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
#     links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

#     albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
#     albedo_file_output.label = 'Albedo Output'
#     albedo_file_output.base_path = output_dir
#     albedo_file_output.file_slots[0].use_node_format = True
#     albedo_file_output.format.file_format = "PNG"
#     albedo_file_output.format.color_mode = 'RGBA'
#     albedo_file_output.format.color_depth ='8'
#     links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])

#     # Create mask map output nodes
#     mask_file_output = nodes.new(type="CompositorNodeOutputFile")
#     mask_file_output.label = 'ID Output'
#     mask_file_output.base_path = output_dir
#     mask_file_output.file_slots[0].use_node_format = True
#     mask_file_output.format.file_format = "PNG"
#     mask_file_output.format.color_depth ='8'
#     mask_file_output.format.color_mode = 'BW'

#     # divide_node = nodes.new(type='CompositorNodeMath')
#     # divide_node.operation = 'DIVIDE'
#     # divide_node.use_clamp = False
#     # divide_node.inputs[1].default_value = 2**int(8)

#     id_mask_node = nodes.new(type='CompositorNodeIDMask')
#     id_mask_node.index = 1  
#     # idmask_node.inputs[1].default_value = 2**int(8)
#     id_mask_node.use_antialiasing = True

#     # links.new(render_layers.outputs['IndexOB'], id_mask_node.inputs[0])
#     links.new(render_layers.outputs['IndexOB'], id_mask_node.inputs['ID value'])
#     links.new(id_mask_node.outputs['Alpha'], mask_file_output.inputs[0])

#     # normalize the scene

#     # normalize_scene()

#     add_wall_background()

#     # # randomize the lighting
#     # randomize_lighting()

#     # File path
#     filename = '/fs/nexus-scratch/sjxu/objaverse-xl/scripts/rendering/meta_unique.txt'  # Replace with your text file name

#     # Read the file into a pandas DataFrame
#     df = pd.read_csv(filename, sep='\t')

#     # Choose ONE random lighting setup
#     row = df.sample(1).iloc[0]
#     light_type = row['light_type']
#     color = ast.literal_eval(row['color'])
#     rotation = ast.literal_eval(row['rotation'])

#     # Set the lighting once
#     randomize_lighting_pd(light_type, color, rotation)

#     radius = 4.0
#     center = Vector((0.0, 0.0, 0.0))  # Assuming object is centered
    
#     stepsize =  1/2 * math.pi / num_renders #args.views
#     rotation_mode = 'XYZ'
    
#     # render the images
#     for i in range(num_renders):
#         # for index, row in df.iterrows():
#         # light_type = row['light_type']
#         # color = ast.literal_eval(row['color'])
#         # rotation = ast.literal_eval(row['rotation'])

#         # # randomize the lighting
#         # randomize_lighting_pd(light_type, color, rotation)

#         # # set camera
#         # camera = randomize_camera(
#         #     only_northern_hemisphere=only_northern_hemisphere,
#         # )
            
#         # angle = 1/2 * math.pi * i / num_renders
#         angle = stepsize * i
#         cam_x = radius * math.cos(angle)
#         cam_y = radius * math.sin(angle)
#         cam_z = 1.0  # Slight elevation

#         cam.location = Vector((cam_x, cam_y, cam_z))

#         # Point the camera at the center
#         direction = (center - cam.location).normalized()
#         rot_quat = direction.to_track_quat("-Z", "Y")
#         cam.rotation_euler = rot_quat.to_euler()

#         # # Render frame
#         # render_path = os.path.join(output_dir, f"{i:03d}.png")
#         # scene.render.filepath = render_path
#         # bpy.ops.render.render(write_still=True)


#         # # Save camera RT matrix
#         # rt_matrix = get_3x4_RT_matrix_from_blender(cam)
#         # rt_matrix_path = os.path.join(output_dir, f"{i:03d}.npy")
#         # np.save(rt_matrix_path, rt_matrix)

#         # Render frame (RGB)
#         render_path = os.path.join(output_dir, '{0:03d}'.format(int(i)))

#         scene.render.filepath = render_path

#         # tmp_base_name = f"tmp_{i:03d}"

#         depth_file_output.file_slots[0].path = render_path + "_depth"
#         normal_file_output.file_slots[0].path = render_path + "_normal"
#         albedo_file_output.file_slots[0].path = render_path + "_albedo"
#         mask_file_output.file_slots[0].path = render_path + "_mask"

#         # # Temporary file paths for compositor nodes (Blender auto-adds frame number)
#         # depth_file_output.file_slots[0].path = tmp_base_name + "_depth"
#         # normal_file_output.file_slots[0].path = tmp_base_name + "_normal"
#         # albedo_file_output.file_slots[0].path = tmp_base_name + "_albedo"
#         # mask_file_output.file_slots[0].path = tmp_base_name + "_mask"

#         print(render_path + "_depth")
#         print(render_path + "_mask")

#         bpy.ops.render.render(write_still=True)  # render still

#         # # Rename temporary outputs to final filenames without frame numbers
#         # for suffix in ["_depth", "_normal", "_albedo", "_mask"]:
#         #     src = os.path.join(output_dir, f"{tmp_base_name}{suffix}0001.png")
#         #     dst = os.path.join(output_dir, f"{i:03d}{suffix}.png")
#         #     if os.path.exists(src):
#         #         shutil.move(src, dst)

#         # Save camera RT matrix
#         rt_matrix = get_3x4_RT_matrix_from_blender(cam)
#         rt_matrix_path = os.path.join(output_dir, f"{i:03d}.npy")
#         np.save(rt_matrix_path, rt_matrix)

    # Create a biased random point between center and spotlight
    # This will be closer to the spotlight based on the random values
    center_point = Vector((0.0, 0.0, 0.0))
    spotlight_point = Vector((spotlight_x, spotlight_y, 0.0))

    # Random weight - how much we want to bias toward spotlight vs center
    # Higher values (closer to 1.0) will bias toward spotlight
    bias_toward_spotlight = random.uniform(0.2, 0.8)  # Adjust these bounds as needed

    # Calculate interpolated position
    interpolated_position = center_point.lerp(spotlight_point, bias_toward_spotlight)

    # Add a small random offset for variation
    random_offset_x = random.uniform(-0.2, 0.2)  # Adjust range as needed
    random_offset_y = random.uniform(-0.2, 0.2)  # Adjust range as needed

    final_position = Vector((
        interpolated_position.x + random_offset_x,
        interpolated_position.y + random_offset_y,
        0.0  # Keeping z at zero
    ))

    print('Final look-at position:', final_position)
    empty.location = final_position
    cam_constraint.target = empty

    # Set current frame to 0 before rendering
    bpy.context.scene.frame_set(0)

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

   # Clear existing nodes
    nodes.clear()

    # Create input render layer node
    render_layers = nodes.new('CompositorNodeRLayers')
    render_layers.location = (0, 0)

    # --- DEPTH BRANCH ---
    # Normalize node
    normalize = nodes.new(type="CompositorNodeNormalize")
    normalize.location = (250, 400)

    # Depth output node
    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = output_dir
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = "PNG"
    depth_file_output.format.color_depth = '8'
    depth_file_output.format.color_mode = "BW"
    depth_file_output.location = (450, 400)

    # Connect depth branch
    links.new(render_layers.outputs['Depth'], normalize.inputs[0])
    links.new(normalize.outputs[0], depth_file_output.inputs[0])

    # --- NORMAL BRANCH ---
    # Separate XYZ node to split normal vector components
    separate_normal = nodes.new(type="CompositorNodeSeparateXYZ")
    separate_normal.location = (250, 0)

    # --- Row 1 of camera matrix (X output) ---
    cam_x_x = nodes.new(type="CompositorNodeMath")
    cam_x_x.operation = 'MULTIPLY'
    cam_x_x.location = (450, 80)

    cam_y_x = nodes.new(type="CompositorNodeMath")
    cam_y_x.operation = 'MULTIPLY'
    cam_y_x.location = (450, 40)

    cam_z_x = nodes.new(type="CompositorNodeMath")
    cam_z_x.operation = 'MULTIPLY'
    cam_z_x.location = (450, 0)

    cam_sum_x = nodes.new(type="CompositorNodeMath")
    cam_sum_x.operation = 'ADD'
    cam_sum_x.location = (650, 60)

    cam_sum_x2 = nodes.new(type="CompositorNodeMath")
    cam_sum_x2.operation = 'ADD'
    cam_sum_x2.location = (800, 40)

    # --- Row 2 of camera matrix (Y output) ---
    cam_x_y = nodes.new(type="CompositorNodeMath")
    cam_x_y.operation = 'MULTIPLY'
    cam_x_y.location = (450, -40)

    cam_y_y = nodes.new(type="CompositorNodeMath")
    cam_y_y.operation = 'MULTIPLY'
    cam_y_y.location = (450, -80)

    cam_z_y = nodes.new(type="CompositorNodeMath")
    cam_z_y.operation = 'MULTIPLY'
    cam_z_y.location = (450, -120)

    cam_sum_y = nodes.new(type="CompositorNodeMath")
    cam_sum_y.operation = 'ADD'
    cam_sum_y.location = (650, -60)

    cam_sum_y2 = nodes.new(type="CompositorNodeMath")
    cam_sum_y2.operation = 'ADD'
    cam_sum_y2.location = (800, -80)

    # --- Row 3 of camera matrix (Z output) ---
    cam_x_z = nodes.new(type="CompositorNodeMath")
    cam_x_z.operation = 'MULTIPLY'
    cam_x_z.location = (450, -160)

    cam_y_z = nodes.new(type="CompositorNodeMath")
    cam_y_z.operation = 'MULTIPLY'
    cam_y_z.location = (450, -200)

    cam_z_z = nodes.new(type="CompositorNodeMath")
    cam_z_z.operation = 'MULTIPLY'
    cam_z_z.location = (450, -240)

    cam_sum_z = nodes.new(type="CompositorNodeMath")
    cam_sum_z.operation = 'ADD'
    cam_sum_z.location = (650, -180)

    cam_sum_z2 = nodes.new(type="CompositorNodeMath")
    cam_sum_z2.operation = 'ADD'
    cam_sum_z2.location = (800, -200)

    # --- Combine and format normal vector ---
    combine_normal = nodes.new(type="CompositorNodeCombineXYZ")
    combine_normal.location = (950, -80)

    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.base_path = output_dir
    normal_file_output.file_slots[0].use_node_format = True
    normal_file_output.format.file_format = "PNG"
    normal_file_output.location = (1400, -80)

    # --- ALBEDO BRANCH ---
    alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
    alpha_albedo.location = (250, -400)

    albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'
    albedo_file_output.base_path = output_dir
    albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = "PNG"
    albedo_file_output.format.color_mode = 'RGBA'
    albedo_file_output.format.color_depth = '8'
    albedo_file_output.location = (450, -400)

    # --- ID MASK BRANCH ---
    id_mask_node = nodes.new(type='CompositorNodeIDMask')
    id_mask_node.index = 1
    id_mask_node.use_antialiasing = True
    id_mask_node.location = (250, -600)

    mask_file_output = nodes.new(type="CompositorNodeOutputFile")
    mask_file_output.label = 'ID Output'
    mask_file_output.base_path = output_dir
    mask_file_output.file_slots[0].use_node_format = True
    mask_file_output.format.file_format = "PNG"
    mask_file_output.format.color_depth = '8'
    mask_file_output.format.color_mode = 'BW'
    mask_file_output.location = (450, -600)

    # --- CONNECTIONS ---
    # Normal branch connections
    links.new(render_layers.outputs['Normal'], separate_normal.inputs[0])

    # X camera coordinate
    links.new(separate_normal.outputs[0], cam_x_x.inputs[0])  # World X
    links.new(separate_normal.outputs[1], cam_y_x.inputs[0])  # World Y
    links.new(separate_normal.outputs[2], cam_z_x.inputs[0])  # World Z
    links.new(cam_x_x.outputs[0], cam_sum_x.inputs[0])
    links.new(cam_y_x.outputs[0], cam_sum_x.inputs[1])
    links.new(cam_sum_x.outputs[0], cam_sum_x2.inputs[0])
    links.new(cam_z_x.outputs[0], cam_sum_x2.inputs[1])

    # Y camera coordinate
    links.new(separate_normal.outputs[0], cam_x_y.inputs[0])  # World X
    links.new(separate_normal.outputs[1], cam_y_y.inputs[0])  # World Y
    links.new(separate_normal.outputs[2], cam_z_y.inputs[0])  # World Z
    links.new(cam_x_y.outputs[0], cam_sum_y.inputs[0])
    links.new(cam_y_y.outputs[0], cam_sum_y.inputs[1])
    links.new(cam_sum_y.outputs[0], cam_sum_y2.inputs[0])
    links.new(cam_z_y.outputs[0], cam_sum_y2.inputs[1])

    # Z camera coordinate
    links.new(separate_normal.outputs[0], cam_x_z.inputs[0])  # World X
    links.new(separate_normal.outputs[1], cam_y_z.inputs[0])  # World Y
    links.new(separate_normal.outputs[2], cam_z_z.inputs[0])  # World Z
    links.new(cam_x_z.outputs[0], cam_sum_z.inputs[0])
    links.new(cam_y_z.outputs[0], cam_sum_z.inputs[1])
    links.new(cam_sum_z.outputs[0], cam_sum_z2.inputs[0])
    links.new(cam_z_z.outputs[0], cam_sum_z2.inputs[1])

    # Connect to combiner and format
    links.new(cam_sum_x2.outputs[0], combine_normal.inputs[0])  # X
    links.new(cam_sum_y2.outputs[0], combine_normal.inputs[1])  # Y
    links.new(cam_sum_z2.outputs[0], combine_normal.inputs[2])  # Z
#    links.new(combine_normal.outputs[0], scale_node.inputs[1])
#    links.new(scale_node.outputs[0], bias_node.inputs[1])
#    links.new(bias_node.outputs[0], normal_file_output.inputs[0])

#    links.new(scale_node.outputs[0], bias_node.inputs[1])
    links.new(combine_normal.outputs[0], normal_file_output.inputs[0])
    
    # Albedo branch connections
    links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
    links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
    links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])

    # ID Mask branch connections
    links.new(render_layers.outputs['IndexOB'], id_mask_node.inputs['ID value'])
    links.new(id_mask_node.outputs['Alpha'], mask_file_output.inputs[0])

    # Set file names for the outputs
    depth_file_output.file_slots[0].path = 'depth_'
    normal_file_output.file_slots[0].path = 'normal_'
    albedo_file_output.file_slots[0].path = 'albedo_'
    mask_file_output.file_slots[0].path = 'id_'

    # Set render to a single frame
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0

    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    add_wall_background()

    # randomize the lighting
#    randomize_lighting()
  
    # # File path
    # filename = '/Users/shengjiexu/Downloads/meta_unique.txt'  # Replace with your text file name

    # # Read the file into a pandas DataFrame
    # df = pd.read_csv(filename, sep='\t')

    # # Choose ONE random lighting setup
    # row = df.sample(1).iloc[0]
    # light_type = row['light_type']
    # color = ast.literal_eval(row['color'])
    # rotation = ast.literal_eval(row['rotation'])

    # Set the lighting once
#    randomize_lighting_pd(light_type, color, rotation)
    lighting_dict = spotlight_on_object((spotlight_x, spotlight_y, 5), (0,0,0))
    key_light = lighting_dict["key_light"]

    radius = 5.0
    elevation = random.uniform(1, 6,)
    # center = Vector((0.0, 0.0, 0.0))  # Assuming object is centered
    center = final_position

    stepsize =  1/2 * math.pi /4 / num_renders #args.views
    rotation_mode = 'XYZ'

    # Store references to nodes that need updating with each camera change
    matrix_nodes = {
        "x_x": cam_x_x,
        "y_x": cam_y_x,
        "z_x": cam_z_x,
        "x_y": cam_x_y,
        "y_y": cam_y_y,
        "z_y": cam_z_y,
        "x_z": cam_x_z,
        "y_z": cam_y_z,
        "z_z": cam_z_z
    }

    # render the images
    for i in range(num_renders):
        timing_info = {'frame': i}

        angle = stepsize * i
        cam_x = radius * math.cos(angle)
        cam_y = radius * math.sin(angle)
        cam_z = elevation  # Slight elevation

        cam.location = Vector((cam_x, cam_y, cam_z))

       # Save camera RT matrix
        rt_matrix = get_3x4_RT_matrix_from_blender(cam)
        rt_matrix_path = os.path.join(output_dir, f"{i:03d}.npy")
        np.save(rt_matrix_path, rt_matrix)
        print(rt_matrix)
        

        # Set the matrix values in the nodes
        matrix_nodes["x_x"].inputs[1].default_value = rt_matrix[0][0]  # R_00
        matrix_nodes["y_x"].inputs[1].default_value = rt_matrix[0][1]  # R_01
        matrix_nodes["z_x"].inputs[1].default_value = rt_matrix[0][2]  # R_02

        matrix_nodes["x_y"].inputs[1].default_value = rt_matrix[1][0]  # R_10
        matrix_nodes["y_y"].inputs[1].default_value = rt_matrix[1][1]  # R_11
        matrix_nodes["z_y"].inputs[1].default_value = rt_matrix[1][2]  # R_12

        matrix_nodes["x_z"].inputs[1].default_value = rt_matrix[2][0]  # R_20
        matrix_nodes["y_z"].inputs[1].default_value = rt_matrix[2][1]  # R_21
        matrix_nodes["z_z"].inputs[1].default_value = rt_matrix[2][2]  # R_22


        # Point the camera at the center
        direction = (center - cam.location).normalized()
        print(center)
        rot_quat = direction.to_track_quat("-Z", "Y")
        cam.rotation_euler = rot_quat.to_euler()

        # Set paths for rendering
        render_filename = f"{i:03d}.png"
        render_filepath = os.path.join(output_dir, render_filename)

        scene.render.filepath = render_filepath

        tmp_base_name = f"tmp_{i:03d}"

        # Temporary file paths for compositor nodes (Blender auto-adds frame number)
        # relit_output.file_slots[0].path = tmp_base_name + "_relit"
        # input_output.file_slots[0].path = tmp_base_name + "_input"
        depth_file_output.file_slots[0].path = tmp_base_name + "_depth"
        normal_file_output.file_slots[0].path = tmp_base_name + "_normal"
        albedo_file_output.file_slots[0].path = tmp_base_name + "_albedo"
        mask_file_output.file_slots[0].path = tmp_base_name + "_mask"

        # Render the frame
        start = time.time()

        bpy.ops.render.render(write_still=True)
        end = time.time()
        timing_info['relit_time_sec'] = round(end - start, 3)

        # Rename temporary outputs to final filenames without frame numbers
        for suffix in ["_depth", "_normal", "_albedo", "_mask"]:
            src = os.path.join(output_dir, f"{tmp_base_name}{suffix}0000.png")
            dst = os.path.join(output_dir, f"{i:03d}{suffix}.png")
            if os.path.exists(src):
                shutil.move(src, dst)
                
        src_relit = os.path.join(output_dir, f"{i:03d}.png")
        dst_relit = os.path.join(output_dir, f"{i:03d}_relit.png")
        shutil.move(src_relit, dst_relit)



        # Now render only the RGB pass with key_light OFF (input version)
        # Disable the other render passes temporarily to avoid re-rendering
        
#        render_layers.mute = True
        depth_file_output.mute = True
        normal_file_output.mute = True
        albedo_file_output.mute = True
        mask_file_output.mute = True

        key_light.hide_render = True

        start = time.time()
        bpy.ops.render.render(write_still=True)
        end = time.time()
        timing_info['input_time_sec'] = round(end - start, 3)

        src_relit = os.path.join(output_dir, f"{i:03d}.png")
        dst_relit = os.path.join(output_dir, f"{i:03d}_input.png")
        shutil.move(src_relit, dst_relit)

        # Re-enable the output nodes for the next iteration
#        render_layers.mute = False
        depth_file_output.mute = False
        normal_file_output.mute = False
        albedo_file_output.mute = False
        mask_file_output.mute = False
        
        key_light.hide_render = False
        timings.append(timing_info)

    df = pd.DataFrame(timings)
    summary_csv_path = os.path.join(output_dir, "render_timing_summary.csv")
    df.to_csv(summary_csv_path, index=False)
    print(f"Saved render timing summary to {summary_csv_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        help="Path to the object file",
    )
    parser.add_argument(
        "--object_paths",
        nargs="+",                    # ← space‑separated list
        # type=list,
        help="Path to the objects file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="BLENDER_EEVEE",
        choices=["CYCLES", "BLENDER_EEVEE"],
    )
    parser.add_argument(
        "--only_northern_hemisphere",
        action="store_true",
        help="Only render the northern hemisphere of the object.",
        default=False,
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=12,
        help="Number of renders to save of the object.",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    context = bpy.context
    scene = context.scene
    render = scene.render

    # Set render settings
    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.color_depth ='8' # ('8', '16')
    render.resolution_x = 512
    render.resolution_y = 512
    render.resolution_percentage = 100

    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_normal = True
    scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
    scene.view_layers["ViewLayer"].use_pass_object_index = True

    # Set cycles settings
    scene.cycles.device = "GPU"
    scene.cycles.samples = 64 #128
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 128
    scene.render.film_transparent = True
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA"  # or "OPENCL"

    # Render the images
    render_object(
        object_file=args.object_path,
        objects_list=args.object_paths,
        num_renders=args.num_renders,
        only_northern_hemisphere=args.only_northern_hemisphere,
        output_dir=args.output_dir,
    )

    # Set output directory and expected modalities
    output_dir = args.output_dir
    modalities = ['input', 'relit', 'depth', 'normal', 'albedo', 'mask']
    # Create a GIF for each modality using the rendered frames
    for modality in modalities:
        images = []
        for i in range(args.num_renders):  # Assuming 4 renders (i=0 to 3)
            frame_path = os.path.join(output_dir, f"{i:03d}_{modality}.png")
            if os.path.exists(frame_path):
                images.append(imageio.imread(frame_path))
    
        if images:
            gif_path = os.path.join(output_dir, f"{modality}.gif")
            imageio.mimsave(gif_path, images, duration=0.5)  # 0.5 seconds per frame
