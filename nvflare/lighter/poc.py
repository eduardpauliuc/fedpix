# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import shutil


def clone_client(num_clients: int, dest_poc_folder):
    src_folder = os.path.join(dest_poc_folder, "client")
    for index in range(1, num_clients + 1):
        dst_folder = os.path.join(dest_poc_folder, f"site-{index}")
        shutil.copytree(src_folder, dst_folder)
        start_sh = open(os.path.join(dst_folder, "startup", "start.sh"), "rt")
        content = start_sh.read()
        start_sh.close()
        content = content.replace("NNN", f"{index}")
        with open(os.path.join(dst_folder, "startup", "start.sh"), "wt") as f:
            f.write(content)
    shutil.rmtree(src_folder)


def unpack_poc(dest_poc_folder) -> bool:
    file_dir_path = pathlib.Path(__file__).parent.absolute()
    poc_zip_path = file_dir_path.parent / "poc.zip"
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.unpack_archive(poc_zip_path, extract_dir=tmp_dir)
            copy_from_src(os.path.join(tmp_dir, "poc"), dest_poc_folder)

        return True
    except shutil.ReadError:
        return False


def copy_from_src(src_poc_folder, dest_poc_folder):
    try:
        shutil.copytree(src_poc_folder, dest_poc_folder)
    except BaseException as e:
        print(f"Unable to copy poc folder from {src_poc_folder}, Exit. {e} ")
        exit(1)


def clone_poc_folder(src_poc_folder, dest_poc_folder):
    shutil.rmtree(dest_poc_folder, ignore_errors=True)
    success = unpack_poc(dest_poc_folder)
    if not success:
        copy_from_src(src_poc_folder, dest_poc_folder)

    for root, dirs, files in os.walk(dest_poc_folder):
        for dir in dirs:
            if dir == "admin":
                try:
                    os.mkdir(os.path.join(root, dir, "local"))
                except BaseException:
                    pass
                break
        for file in files:
            if file.endswith(".sh"):
                os.chmod(os.path.join(root, file), 0o755)


def get_src_poc_dir():
    file_dir_path = pathlib.Path(__file__).parent.absolute()
    parent = file_dir_path.parent
    return os.path.join(parent, "poc")


def get_dest_poc_dir(dest_poc):
    if not dest_poc:
        dest_poc = os.path.join(os.getcwd(), "poc")
    return dest_poc


def generate_poc(num_clients, poc_workspace):
    answer = input(
        f"This will delete poc folder in {poc_workspace} directory and create a new one. Is it OK to proceed? (y/N) "
    )
    if answer.strip().upper() == "Y":
        dest_poc_folder = get_dest_poc_dir(poc_workspace)
        src_poc_folder = get_src_poc_dir()
        clone_poc_folder(src_poc_folder=src_poc_folder, dest_poc_folder=dest_poc_folder)
        clone_client(num_clients, dest_poc_folder)
        print(f"Successfully creating poc folder at {dest_poc_folder}.  Please read poc/Readme.rst for user guide.")
        print("\n\nWARNING:\n******* Files generated by this poc command are NOT intended for production environments.")
        return True
    else:
        return False


def main():
    print("*****************************************************************")
    print("** poc command is deprecated, please use 'nvflare poc' instead **")
    print("*****************************************************************")


if __name__ == "__main__":
    main()
