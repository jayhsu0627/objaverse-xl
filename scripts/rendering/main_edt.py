import glob
import json
import multiprocessing
import os
import platform
import random
import subprocess
import tempfile
import time
import zipfile
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Union, Tuple
import shlex, math

import fire
import fsspec
import GPUtil
import pandas as pd
from loguru import logger

import objaverse.xl as oxl
import xl as mod_oxl
from objaverse.utils import get_uid_from_str, get_file_hash
from objaverse import load_uids, load_objects, load_lvis_annotations  # same helper you already call
from objaverse.xl import download_objects, SketchfabDownloader   # you already vendor this
from multiprocessing import Manager   # add to imports at top of file

import time
import datetime

def log_processed_object(csv_filename: str, *args) -> None:
    """Log when an object is done being used.

    Args:
        csv_filename (str): Name of the CSV file to save the logs to.
        *args: Arguments to save to the CSV file.

    Returns:
        None
    """
    args = ",".join([str(arg) for arg in args])
    # log that this object was rendered successfully
    # saving locally to avoid excessive writes to the cloud
    # dirname = os.path.expanduser(f"~/.objaverse/logs/")
    dirname = os.path.expanduser(f"/fs/nexus-scratch/sjxu/.objaverse/logs/")
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, csv_filename), "a", encoding="utf-8") as f:
        f.write(f"{time.time()},{args}\n")


def zipdir(path: str, ziph: zipfile.ZipFile) -> None:
    """Zip up a directory with an arcname structure.

    Args:
        path (str): Path to the directory to zip.
        ziph (zipfile.ZipFile): ZipFile handler object to write to.

    Returns:
        None
    """
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            # this ensures the structure inside the zip starts at folder/
            arcname = os.path.join(os.path.basename(root), file)
            ziph.write(os.path.join(root, file), arcname=arcname)

def get_and_rename_files(directory_path):
    # Get all files in the directory that end with '0001.png'
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('0001.png'):
                all_files.append(os.path.join(root, file))
    
    # Rename each file
    renamed_count = 0
    for file_path in all_files:
        # Create the new file name by replacing '0001.png' with '.png'
        new_file_path = file_path.replace('0001.png', '.png')
        
        # Rename the file
        try:
            os.rename(file_path, new_file_path)
            renamed_count += 1
            # print(f"Renamed: {file_path} -> {new_file_path}")
        except Exception as e:
            print(f"Error renaming {file_path}: {e}")
    
    # print(f"\nTotal files renamed: {renamed_count}")
    return renamed_count

# def handle_found_object(
#     local_path: str,
#     file_identifier: str,
#     sha256: str,
#     metadata: Dict[str, Any],
#     num_renders: int,
#     render_dir: str,
#     only_northern_hemisphere: bool,
#     gpu_devices: Union[int, List[int]],
#     render_timeout: int,
#     # ↓ NEW shared state objects injected by `partial` in render_objects()
#     basket,                 # multiprocessing.Manager().list()
#     target_size,            # multiprocessing.Manager().Value('i', …)
#     scene_counter,          # multiprocessing.Manager().Value('i', …)
#     max_scenes: int,
#     min_objs: int,
#     max_objs: int,
#     successful_log_file: Optional[str] = "handle-found-object-successful.csv",
#     failed_log_file: Optional[str] = "handle-found-object-failed.csv",
# ) -> bool:
#     """Called when an object is successfully found and downloaded.

#     Here, the object has the same sha256 as the one that was downloaded with
#     Objaverse-XL. If None, the object will be downloaded, but nothing will be done with
#     it.

#     Args:
#         local_path (str): Local path to the downloaded 3D object.
#         file_identifier (str): File identifier of the 3D object.
#         sha256 (str): SHA256 of the contents of the 3D object.
#         metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
#             organization and repo names.
#         num_renders (int): Number of renders to save of the object.
#         render_dir (str): Directory where the objects will be rendered.
#         only_northern_hemisphere (bool): Only render the northern hemisphere of the
#             object.
#         gpu_devices (Union[int, List[int]]): GPU device(s) to use for rendering. If
#             an int, the GPU device will be randomly selected from 0 to gpu_devices - 1.
#             If a list, the GPU device will be randomly selected from the list.
#             If 0, the CPU will be used for rendering.
#         render_timeout (int): Number of seconds to wait for the rendering job to
#             complete.
#         successful_log_file (str): Name of the log file to save successful renders to.
#         failed_log_file (str): Name of the log file to save failed renders to.

#     Returns: True if the object was rendered successfully, False otherwise.
   
#     Queues each downloaded object until `len(basket) == target_size.value`,
#     then renders *all* of them together in one Blender scene.  Stops after
#     `max_scenes` scenes have been rendered.

#     The function keeps **all** of your original post‑processing (renaming PNGs,
#     zipping, uploading, logging) so downstream code remains unchanged.
#     """
#     print('here')
#     # 0️⃣  Add the freshly‑downloaded file to the shared basket
#     basket.append(local_path)
#     if len(basket) < target_size.value:
#         return False                        # tell download_objects we’re not done

#     if scene_counter.value >= max_scenes:   # hard stop guard
#         return True                         # skip further processing

#     # ------------------------------------------------------------------
#     # 1️⃣  Build Blender CLI args for every object in the basket
#     # joined_paths = " ".join(f"'{p}'" for p in basket)   # quote in case of spaces
#     quoted = " ".join(shlex.quote(p) for p in basket)
#     args = f"--object_paths {quoted} --num_renders {num_renders}"

#     # # joined_paths = basket # a list of path
#     # args = f"--object_paths {joined_paths} --num_renders {num_renders}"

#     # save_uid = get_uid_from_str(file_identifier)
#     # args = f"--object_path '{local_path}' --num_renders {num_renders}"

#     # get the GPU to use for rendering
#     using_gpu: bool = True
#     gpu_i = 0
#     if isinstance(gpu_devices, int) and gpu_devices > 0:
#         num_gpus = gpu_devices
#         gpu_i = random.randint(0, num_gpus - 1)
#     elif isinstance(gpu_devices, list):
#         gpu_i = random.choice(gpu_devices)
#     elif isinstance(gpu_devices, int) and gpu_devices == 0:
#         using_gpu = False
#     else:
#         raise ValueError(
#             f"gpu_devices must be an int > 0, 0, or a list of ints. Got {gpu_devices}."
#         )
    
#     # 2️⃣  Run Blender once, importing *all* objects
#     save_uid = f"scene_{scene_counter.value:04d}"
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # get the target directory for the rendering job
#         target_directory = os.path.join(temp_dir, save_uid)
#         os.makedirs(target_directory, exist_ok=True)
#         args += f" --output_dir {target_directory}"

#         # check for Linux / Ubuntu or MacOS
#         if platform.system() == "Linux" and using_gpu:
#             # args += " --engine BLENDER_EEVEE"
#             args += " --engine CYCLES"
#         elif platform.system() == "Darwin" or (
#             platform.system() == "Linux" and not using_gpu
#         ):
#             # As far as I know, MacOS does not support BLENER_EEVEE, which uses GPU
#             # rendering. Generally, I'd only recommend using MacOS for debugging and
#             # small rendering jobs, since CYCLES is much slower than BLENDER_EEVEE.
#             args += " --engine CYCLES"
#         else:
#             raise NotImplementedError(f"Platform {platform.system()} is not supported.")

#         # check if we should only render the northern hemisphere
#         if only_northern_hemisphere:
#             args += " --only_northern_hemisphere"

#         # get the command to run
#         # command = f"blender-3.2.2-linux-x64/blender --background --python blender_script.py -- {args}"
#         # command = f"blender-3.2.2-linux-x64/blender --background --python blender_script_cp.py {args}"
#         command = f"blender-3.2.2-linux-x64/blender -b --python blender_script_cp.py -- {args}"

#         # command = f"xvfb-run -s \"-screen 0 1024x768x24 -ac +extension GLX +render -noreset\" blender-3.2.2-linux-x64/blender --background --python blender_script.py -- {args}"
#         # command = f"xvfb-run -s \"-screen 0 1024x768x24\" blender-3.2.2-linux-x64/blender --background --python blender_script_cp.py -- {args}"

#         # https://devtalk.blender.org/t/blender-2-8-unable-to-open-a-display-by-the-rendering-on-the-background-eevee/1436/10
#         # https://yigityakupoglu.home.blog/
        
#         if using_gpu:
#             command = f"export DISPLAY=:0.{gpu_i} && {command}"
        
#         # # render the object (put in dev null)
#         # subprocess.run(
#         #     ["bash", "-c", command],
#         #     timeout=render_timeout,
#         #     check=False,
#         #     stdout=subprocess.DEVNULL,
#         #     stderr=subprocess.DEVNULL,
#         # )

#         result = subprocess.run(
#             ["bash", "-c", command],
#             timeout=render_timeout,
#             capture_output=True,
#             text=True
#         )

#         # Save GPU usage info from stdout
#         logger.info(f"[{file_identifier}] Blender stdout:\n{result.stdout}")

#         if result.returncode != 0:
#             logger.error(f"[{file_identifier}] Blender render failed:\n{result.stderr}")

#         # 3️⃣  Your original post‑processing (rename PNGs, zip, upload, log)

#         # check that the renders were saved successfully
#         # [os.rename(f, f.replace('0001.png', '.png')) for f in glob.glob(os.path.join(target_directory, "*.png"))]
#         get_and_rename_files(target_directory)

#         png_files = glob.glob(os.path.join(target_directory, "*.png"))
#         metadata_files = glob.glob(os.path.join(target_directory, "*.json"))
#         npy_files = glob.glob(os.path.join(target_directory, "*.npy"))
        
#         if (
#             (len(png_files) != num_renders)
#             or (len(npy_files) != num_renders)
#             or (len(metadata_files) != 1)
#         ):
#             logger.error(
#                 f"Found object {file_identifier} was not rendered successfully!"
#             )
#             if failed_log_file is not None:
#                 log_processed_object(
#                     failed_log_file,
#                     file_identifier,
#                     sha256,
#                 )
#             return False

#         # update the metadata
#         metadata_path = os.path.join(target_directory, "metadata.json")
#         with open(metadata_path, "r", encoding="utf-8") as f:
#             metadata_file = json.load(f)
#         metadata_file["sha256"] = sha256
#         metadata_file["file_identifier"] = file_identifier
#         metadata_file["save_uid"] = save_uid
#         metadata_file["metadata"] = metadata
#         with open(metadata_path, "w", encoding="utf-8") as f:
#             json.dump(metadata_file, f, indent=2, sort_keys=True)

#         # Make a zip of the target_directory.
#         # Keeps the {save_uid} directory structure when unzipped
#         with zipfile.ZipFile(
#             f"{target_directory}.zip", "w", zipfile.ZIP_DEFLATED
#         ) as ziph:
#             zipdir(target_directory, ziph)

#         # move the zip to the render_dir
#         fs, path = fsspec.core.url_to_fs(render_dir)

#         # move the zip to the render_dir
#         fs.makedirs(os.path.join(path, "renders"), exist_ok=True)
#         fs.put(
#             os.path.join(f"{target_directory}.zip"),
#             os.path.join(path, "renders", f"{save_uid}.zip"),
#         )

#         # log that this object was rendered successfully
#         if successful_log_file is not None:
#             log_processed_object(successful_log_file, file_identifier, sha256)


#     # ------------------------------------------------------------------
#     # 4️⃣  Prepare for the next scene
#     basket[:] = []                                    # empty list in‑place
#     target_size.value = random.randint(min_objs, max_objs)
#     scene_counter.value += 1

#     return True

def handle_found_object(
    local_path: str,
    file_identifier: Optional[str],
    sha256: Optional[str],
    metadata: Optional[Dict[str, Any]],
    num_renders: int,
    render_dir: str,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
    # Shared state objects
    basket,
    target_size,
    scene_counter,
    max_scenes: int,
    min_objs: int,
    max_objs: int,
    successful_log_file: Optional[str] = "handle-found-object-successful.csv",
    failed_log_file: Optional[str] = "handle-found-object-failed.csv",
) -> bool:
    """
    Modified to collect objects until we have enough for a scene, then render them together.
    Returns False to keep downloading more objects until we've rendered max_scenes.
    """
    # # Verify the downloaded file exists before adding to basket
    # print('local_path', local_path)
    # if not os.path.exists(local_path):
    #     logger.error(f"File {local_path} does not exist for {file_identifier}")
    #     if failed_log_file is not None:
    #         log_processed_object(
    #             failed_log_file,
    #             file_identifier,
    #             sha256,
    #             "File not found"
    #         )
    #     return False  # Continue processing other objects

    # Add the freshly‑downloaded file to the shared basket
    if len(basket)< target_size.value:
        basket.append(local_path)
    else:
        basket[:] = []  # Empty the list in-place
    print(basket)

    # for (file,_,_,_) in basket:
    #     print(file, os.path.exists(file))
    for path in basket:
        if not os.path.isfile(path):
            print(f"Missing or invalid file: {path}")

    # print(basket, len(basket))

    # print(len(basket), target_size.value, scene_counter.value >= max_scenes)
    # If we don't have enough objects yet, keep collecting
    if len(basket) < target_size.value:
        return False  # Tell download_objects we're not done, keep downloading
    
    # If we've reached our max scenes limit, stop processing
    if scene_counter.value >= max_scenes:
        return True  # Tell download_objects we're done
    
    print(len(basket), target_size.value)
    print(len(basket) < target_size.value)
    
    # We have enough objects for a scene, process them
    try:
        process_object_batch(
            basket=basket,
            num_renders=num_renders,
            render_dir=render_dir,
            only_northern_hemisphere=only_northern_hemisphere,
            gpu_devices=gpu_devices,
            render_timeout=render_timeout,
            scene_counter=scene_counter,
            successful_log_file=successful_log_file,
            failed_log_file=failed_log_file
        )
    except Exception as e:
        logger.error(f"Error processing batch: {e}")

    for path in basket:
        if os.path.isfile(path):
            os.remove(path)
            print(f"Deleted: {path}")
        else:
            print(f"File not found: {path}")

    # Clear the basket and prepare for the next scene
    target_size.value = 5
    
    print('debugging here', scene_counter.value)
    print(scene_counter.value >= max_scenes)

    # Return False to keep downloading more objects (unless we've hit max_scenes)
    return scene_counter.value >= max_scenes


def process_object_batch(
    basket,
    num_renders: int,
    render_dir: str,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
    scene_counter,
    successful_log_file: Optional[str] = "handle-found-object-successful.csv",
    failed_log_file: Optional[str] = "handle-found-object-failed.csv",
) -> bool:
    """
    Process a batch of objects to render them in a single Blender scene.
    This contains the main rendering logic separated from the collection logic.
    """
    print('==== CALLED ===')
    scene_counter.value += 1
    # Extract paths for the Blender command
    paths = [item for item in basket]  # local_path is first item in each tuple
    quoted = " ".join(shlex.quote(p) for p in paths)
    print("--object_paths", quoted)

    args = f"--object_paths {quoted} --num_renders {num_renders}"

    # Get GPU to use for rendering
    using_gpu: bool = True
    gpu_i = 0
    if isinstance(gpu_devices, int) and gpu_devices > 0:
        num_gpus = gpu_devices
        gpu_i = random.randint(0, num_gpus - 1)
    elif isinstance(gpu_devices, list):
        gpu_i = random.choice(gpu_devices)
    elif isinstance(gpu_devices, int) and gpu_devices == 0:
        using_gpu = False
    else:
        raise ValueError(
            f"gpu_devices must be an int > 0, 0, or a list of ints. Got {gpu_devices}."
        )
    
    # Run Blender once, importing all objects
    save_uid = f"scene_{scene_counter.value:04d}"
    print('save_uid', save_uid)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get the target directory for the rendering job
        target_directory = os.path.join(temp_dir, save_uid)
        os.makedirs(target_directory, exist_ok=True)
        args += f" --output_dir {target_directory}"

        # Check for Linux / Ubuntu or MacOS
        if platform.system() == "Linux" and using_gpu:
            args += " --engine CYCLES"
        elif platform.system() == "Darwin" or (
            platform.system() == "Linux" and not using_gpu
        ):
            args += " --engine CYCLES"
        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

        # Check if we should only render the northern hemisphere
        if only_northern_hemisphere:
            args += " --only_northern_hemisphere"

        # Get the command to run
        command = f"blender-3.2.2-linux-x64/blender -b --python blender_script_cp.py -- {args}"
        
        if using_gpu:
            command = f"export DISPLAY=:0.{gpu_i} && {command}"
        
        # Run the command
        result = subprocess.run(
            ["bash", "-c", command],
            timeout=render_timeout,
            capture_output=True,
            text=True
        )

        # # Activate for debugging Blender
        # # Save GPU usage info from stdout
        # logger.info(f"[{save_uid}] Blender stdout:\n{result.stdout}")

        if result.returncode != 0:
            logger.error(f"[{save_uid}] Blender render failed:\n{result.stderr}")
            return False

        # Post-processing (rename PNGs, zip, upload, log)
        get_and_rename_files(target_directory)

        png_files = glob.glob(os.path.join(target_directory, "*.png"))
        metadata_files = glob.glob(os.path.join(target_directory, "*.json"))
        npy_files = glob.glob(os.path.join(target_directory, "*.npy"))
        
        if (
            (len(png_files) != num_renders * 6)
            or (len(npy_files) != num_renders)
            # or (len(metadata_files) != 1)
        ):
            logger.error(f"Scene {save_uid} was not rendered successfully!")
            # # Log failed objects
            # if failed_log_file is not None:
            #     for file_id in basket:
            #         log_processed_object(failed_log_file, file_id, sha)
            # return False

        # # Update the metadata
        # metadata_path = os.path.join(target_directory, "metadata.json")
        # with open(metadata_path, "r", encoding="utf-8") as f:
        #     metadata_file = json.load(f)
        
        # # Add metadata for all objects in the scene
        # metadata_file["scene_id"] = save_uid
        # metadata_file["objects"] = []
        # for path in basket:
        #     metadata_file["objects"].append({
        #         "file_identifier": 'abc',
        #         "sha256": '123',
        #         "metadata": None
        #     })
        
        # with open(metadata_path, "w", encoding="utf-8") as f:
        #     json.dump(metadata_file, f, indent=2, sort_keys=True)

        # Make a zip of the target_directory
        with zipfile.ZipFile(
            f"{target_directory}.zip", "w", zipfile.ZIP_DEFLATED
        ) as ziph:
            zipdir(target_directory, ziph)

        # Move the zip to the render_dir
        fs, path = fsspec.core.url_to_fs(render_dir)
        fs.makedirs(os.path.join(path, "renders"), exist_ok=True)
        fs.put(
            os.path.join(f"{target_directory}.zip"),
            os.path.join(path, "renders", f"{save_uid}.zip"),
        )

        # # Log that these objects were rendered successfully
        # if successful_log_file is not None:
        #     for _, file_id, sha, _ in basket:
        #         log_processed_object(successful_log_file, file_id, sha)

    return True

def handle_new_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[str, Any],
    log_file: str = "handle-new-object.csv",
) -> None:
    """Called when a new object is found.

    Here, the object is not used in Objaverse-XL, but is still downloaded with the
    repository. The object may have not been used because it does not successfully
    import into Blender. If None, the object will be downloaded, but nothing will be
    done with it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): The file identifier of the new 3D object.
        sha256 (str): SHA256 of the contents of the 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, including the GitHub
            organization and repo names.
        log_file (str): Name of the log file to save the handle_new_object logs to.

    Returns:
        None
    """
    # log the new object
    log_processed_object(log_file, file_identifier, sha256)


def handle_modified_object(
    local_path: str,
    file_identifier: str,
    new_sha256: str,
    old_sha256: str,
    metadata: Dict[str, Any],
    num_renders: int,
    render_dir: str,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
) -> None:
    """Called when a modified object is found and downloaded.

    Here, the object is successfully downloaded, but it has a different sha256 than the
    one that was downloaded with Objaverse-XL. This is not expected to happen very
    often, because the same commit hash is used for each repo. If None, the object will
    be downloaded, but nothing will be done with it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): File identifier of the 3D object.
        new_sha256 (str): SHA256 of the contents of the newly downloaded 3D object.
        old_sha256 (str): Expected SHA256 of the contents of the 3D object as it was
            when it was downloaded with Objaverse-XL.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        num_renders (int): Number of renders to save of the object.get_example_objects
        render_dir (str): Directory where the objects will be rendered.
        only_northern_hemisphere (bool): Only render the northern hemisphere of the
            object.
        gpu_devices (Union[int, List[int]]): GPU device(s) to use for rendering. If
            an int, the GPU device will be randomly selected from 0 to gpu_devices - 1.
            If a list, the GPU device will be randomly selected from the list.
            If 0, the CPU will be used for rendering.
        render_timeout (int): Number of seconds to wait for the rendering job to
            complete.

    Returns:
        None
    """
    success = handle_found_object(
        local_path=local_path,
        file_identifier=file_identifier,
        sha256=new_sha256,
        metadata=metadata,
        num_renders=num_renders,
        render_dir=render_dir,
        only_northern_hemisphere=only_northern_hemisphere,
        gpu_devices=gpu_devices,
        render_timeout=render_timeout,
        successful_log_file=None,
        failed_log_file=None,
    )

    if success:
        log_processed_object(
            "handle-modified-object-successful.csv",
            file_identifier,
            old_sha256,
            new_sha256,
        )
    else:
        log_processed_object(
            "handle-modified-object-failed.csv",
            file_identifier,
            old_sha256,
            new_sha256,
        )


def handle_missing_object(
    file_identifier: str,
    sha256: str,
    metadata: Dict[str, Any],
    log_file: str = "handle-missing-object.csv",
) -> None:
    """Called when an object that is in Objaverse-XL is not found.

    Here, it is likely that the repository was deleted or renamed. If None, nothing
    will be done with the missing object.

    Args:
        file_identifier (str): File identifier of the 3D object.
        sha256 (str): SHA256 of the contents of the original 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        log_file (str): Name of the log file to save missing renders to.

    Returns:
        None
    """
    # log the missing object
    log_processed_object(log_file, file_identifier, sha256)


def get_example_objects() -> pd.DataFrame:
    """Returns a DataFrame of example objects to use for debugging."""
    return pd.read_json("example-objects.json", orient="records")


def render_objects(
    # render_dir: str = "~/.objaverse",
    render_dir: str = "/fs/nexus-scratch/sjxu/.objaverse",
    download_dir: Optional[str] = None,
    num_renders: int = 16,
    processes: Optional[int] = None,
    save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]] = None,
    only_northern_hemisphere: bool = False,
    render_timeout: int = 1000,
    gpu_devices: Optional[Union[int, List[int]]] = None,
) -> None:
    """Renders objects in the Objaverse-XL dataset with Blender

    Args:
        render_dir (str, optional): Directory where the objects will be rendered.
        download_dir (Optional[str], optional): Directory where the objects will be
            downloaded. If None, the objects will not be downloaded. Defaults to None.
        num_renders (int, optional): Number of renders to save of the object. Defaults
            to 12.
        processes (Optional[int], optional): Number of processes to use for downloading
            the objects. If None, defaults to multiprocessing.cpu_count() * 3. Defaults
            to None.
        save_repo_format (Optional[Literal["zip", "tar", "tar.gz", "files"]], optional):
            If not None, the GitHub repo will be deleted after rendering each object
            from it.
        only_northern_hemisphere (bool, optional): Only render the northern hemisphere
            of the object. Useful for rendering objects that are obtained from
            photogrammetry, since the southern hemisphere is often has holes. Defaults
            to False.
        render_timeout (int, optional): Number of seconds to wait for the rendering job
            to complete. Defaults to 300.
        gpu_devices (Optional[Union[int, List[int]]], optional): GPU device(s) to use
            for rendering. If an int, the GPU device will be randomly selected from 0 to
            gpu_devices - 1. If a list, the GPU device will be randomly selected from
            the list. If 0, the CPU will be used for rendering. If None, all available
            GPUs will be used. Defaults to None.

    Returns:
        None
    """
    
    if platform.system() not in ["Linux", "Darwin"]:
        raise NotImplementedError(
            f"Platform {platform.system()} is not supported. Use Linux or MacOS."
        )
    if download_dir is None and save_repo_format is not None:
        raise ValueError(
            f"If {save_repo_format=} is not None, {download_dir=} must be specified."
        )
    if download_dir is not None and save_repo_format is None:
        logger.warning(
            f"GitHub repos will not save. While {download_dir=} is specified, {save_repo_format=} None."
        )

    # get the gpu devices to use
    parsed_gpu_devices: Union[int, List[int]] = 0
    if gpu_devices is None:
        parsed_gpu_devices = len(GPUtil.getGPUs())
    logger.info(f"Using {parsed_gpu_devices} GPU devices for rendering.")

    if processes is None:
        processes = multiprocessing.cpu_count() * 3

    # # get the objects to render
    # objects = get_example_objects()

    # annotations = oxl.get_annotations(
    #     download_dir="~/.objaverse" # default download directory
    # ) 
    # objects = annotations.sample(5)

    # objects.iloc[0]["fileIdentifier"]
    # objects = objects.copy()
    # logger.info(f"Provided {len(objects)} objects to render.")

    # # get the already rendered objects
    # fs, path = fsspec.core.url_to_fs(render_dir)
    # try:
    #     zip_files = fs.glob(os.path.join(path, "renders", "*.zip"), refresh=True)
    # except TypeError:
    #     # s3fs may not support refresh depending on the version
    #     zip_files = fs.glob(os.path.join(path, "renders", "*.zip"))

    # saved_ids = set(zip_file.split("/")[-1].split(".")[0] for zip_file in zip_files)
    # logger.info(f"Found {len(saved_ids)} objects already rendered.")

    # # filter out the already rendered objects
    # objects["saveUid"] = objects["fileIdentifier"].apply(get_uid_from_str)
    # objects = objects[~objects["saveUid"].isin(saved_ids)]
    # objects = objects.reset_index(drop=True)
    # logger.info(f"Rendering {len(objects)} new objects.")

    # # shuffle the objects
    # objects = objects.sample(frac=1).reset_index(drop=True)

    # Load LVIS Annotations (objaverse-xl)
    sketchfab = SketchfabDownloader()
    lvis = sketchfab.get_annotations()                     # {category: [uid, uid, …]}
    lvis['uid'] = lvis['fileIdentifier'].str.split('/').str[-1]  # Takes the last part of the URL

    # Load LVIS json (objaverse 1.0)
    lvis_json = load_lvis_annotations()
    categories = list(lvis_json.keys())

    # Choose how much objects to pick

    k = random.randint(50, 100)
    # 1️⃣  Choose categories first (no repeats until we run out of categories)
    if k <= len(categories):
        chosen_cats = random.sample(categories, k)     # all unique
    else:
        # Need more objects than categories → allow repeats after exhaustion
        chosen_cats = random.sample(categories, len(categories))
        chosen_cats += [random.choice(categories) for _ in range(k - len(categories))]

    # 2️⃣  Pick one random UID from each chosen category
    chosen_uids = [random.choice(lvis_json[cat]) for cat in chosen_cats]

    print(f"🎲  Selected {k} objects from {len(set(chosen_cats))} categories")
    for cat, uid in zip(chosen_cats, chosen_uids):
        print(f"  • {cat:>20s}  →  {uid}")

    # Filter the annotation according choosen uid
    uid_set = set(chosen_uids)  # Convert to set for O(1) lookups
    selected_lvis = lvis[lvis['uid'].map(lambda x: x in uid_set)]
    # print(selected_lvis)

    # print(selected_lvis.sample(3))
    objects = selected_lvis.sample(30)

    # >>>>>>>>>>>>>>>>>>> NEW CONFIG <<<<<<<<<<<<<<<<<<<<<
    min_objs_per_scene = 2
    max_objs_per_scene = 4
    max_scenes_to_render = 5
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # ------------------------------------------------------------------
    # shared objects for cross‑process coordination
    manager       = Manager()
    basket        = manager.list()                         # holds paths
    target_size   = manager.Value('i', 5)
    scene_counter = manager.Value('i', 0)

    print(target_size, scene_counter, basket)
    download_dir = render_dir

    # print(download_dir)
    # dl_report = oxl.download_objects(objects)
    # # print(dl_report)

    # local_path = []
    # for value in dl_report.values():
    #     local_path.append(value)
    # print('local_path: ', local_path)

    # handle_found_object(
    #     local_path             = local_path,
    #     file_identifier        = 'abc',
    #     sha256                 = '123',
    #     metadata               = None,
    #     num_renders            = num_renders,
    #     render_dir             = render_dir,
    #     only_northern_hemisphere = only_northern_hemisphere,
    #     gpu_devices            = parsed_gpu_devices,
    #     render_timeout         = render_timeout,
    #     # shared state
    #     basket=basket,
    #     target_size=target_size,
    #     scene_counter=scene_counter,
    #     max_scenes=max_scenes_to_render,
    #     min_objs=min_objs_per_scene,
    #     max_objs=max_objs_per_scene,
    #     successful_log_file="handle-found-object-successful.csv",
    #     failed_log_file="handle-found-object-failed.csv",
    # )


    # oxl.download_objects(
    mod_oxl.download_objects(
    # download_objects(
        objects=objects,
        # download_dir = "/fs/nexus-scratch/sjxu/.objaverse",
        processes=processes,
        save_repo_format=save_repo_format,
        download_dir=download_dir,
        handle_found_object=partial(
            handle_found_object,
            render_dir=render_dir,
            num_renders=num_renders,
            only_northern_hemisphere=only_northern_hemisphere,
            gpu_devices=parsed_gpu_devices,
            render_timeout=render_timeout,
            # ---------- pass shared state ----------
            basket=basket,
            target_size=target_size,
            scene_counter=scene_counter,
            max_scenes=max_scenes_to_render,
            min_objs=min_objs_per_scene,
            max_objs=max_objs_per_scene,

        ),
        handle_new_object=handle_new_object,
        handle_modified_object=partial(
            handle_modified_object,
            render_dir=render_dir,
            num_renders=num_renders,
            only_northern_hemisphere=only_northern_hemisphere,
            gpu_devices=parsed_gpu_devices,
            render_timeout=render_timeout,
        ),
        handle_missing_object=handle_missing_object,
    )


if __name__ == "__main__":
    filename = "/fs/nexus-scratch/sjxu/objaverse-xl/scripts/rendering/loop_time_cycle.txt"
    start_time = datetime.datetime.now()
    fire.Fire(render_objects)
    end_time = datetime.datetime.now()
    loop_time = end_time - start_time
    # loop_time = loop_time.seconds//3600
    
    # Calculate hours, minutes, and seconds from loop_time
    total_seconds = int(loop_time.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    # Format as "hours:minutes:seconds"
    loop_time_formatted = f"{hours:02}:{minutes:02}:{seconds:02}"

    # Write the loop time to the file
    with open(filename, "a") as file:
        file.write("Run: "+ loop_time_formatted +"seconds\n")
    # subprocess.run(["bash", "-c", "sudo python3 start_x_server.py stop"])
