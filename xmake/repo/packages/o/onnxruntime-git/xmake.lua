package("onnxruntime-git")
    set_kind("library")
    set_homepage("https://www.onnxruntime.ai")
    set_description("ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator")
    set_license("MIT")
    add_urls("https://github.com/microsoft/onnxruntime/archive/v$(version).tar.gz",
        "https://github.com/microsoft/onnxruntime.git")
    
    add_versions("1.20.1", "d4c005506a2bbf88a838b14f8d1578406b8be2fb64abb50beeff908fb272529e")
    add_versions("1.15.0", "7752defd687138870974aa391bd5d3d16228ee449f408d11ea3716e8585c73b4")

    add_patches("1.20.1", path.join(os.scriptdir(), "patches", "1.20.1_rknpu.patch"), "e40b832a224b1b503f14caa56a820041894b2301920dc3b3a9979d8c0ea33abf")

    --##configs##
    add_configs("shared", { description = "Forced Shared Library", default = true, readonly = true })
    -- training
    add_configs("training_api", {description = "Build with training apis", default = false, type = "boolean"})
    add_configs("lm_training", {description = "Build with Large Model Training", default = false, type = "boolean"})
    -- ep
    add_configs("rknpu", {description = "Build with RKNPU Executor Provider Support.", default = false, type = "boolean"})
    --add_configs("cuda", {description = "Build with CUDA Executor Provider Support.", default = false, type = "boolean"})

    add_deps("cmake","python 3.x", {kind = "binary"})
    on_load(function(package)
        package:add("includedirs", "include/onnxruntime")
        if package:config("rknpu") then
            package:add("deps", "rknpu_ddk")
        end
    end)

    on_install("linux", "macosx", "windows", "mingw@windows,msys", function (package)
        local build_type = package:debug() and "Debug" or "Release"
        local build_cmd = format('./build.sh --config %s --parallel --compile_no_warning_as_error --skip_submodule_sync --allow_running_as_root --skip_tests --build_dir build',build_type)
        local install_to = package:installdir()
        local common_cmake_defines = "onnxruntime_ENABLE_PYTHON=OFF onnxruntime_ENABLE_CPUINFO=OFF onnxruntime_BUILD_SHARED_LIB=ON onnxruntime_DEV_MODE=OFF onnxruntime_GCC_STATIC_CPP_RUNTIME=ON protobuf_BUILD_SHARED_LIBS=OFF protobuf_WITH_ZLIB=OFF CMAKE_INSTALL_PREFIX="..install_to
        if package:config("training_api") then
            common_cmake_defines = common_cmake_defines.." --enable_training_apis --enable_training_ops"
        end
        if package:config("lm_training") then
            common_cmake_defines = common_cmake_defines.." --enable_training"
        end
        if package:is_plat("linux") then
            if is_arch("arm64") then
                -- TODO:
                -- 1. fix compile toolchain
                local toolchain_file_path = path.join(package:scriptdir(), "cross-cmake","arm64.toolchain.cmake")
                build_cmd = build_cmd.." --arm64"
                if package:config("rknpu") then
                    local rknpu_ddk_path = package:dep("rknpu_ddk"):installdir()
                    build_cmd = build_cmd.." --use_rknpu --cmake_extra_defines "..common_cmake_defines.." CMAKE_TOOLCHAIN_FILE="..toolchain_file_path.." RKNPU_DDK_PATH="..rknpu_ddk_path
                else
                    build_cmd = build_cmd.." --cmake_extra_defines "..common_cmake_defines.." CMAKE_TOOLCHAIN_FILE="..toolchain_file_path
                end
            elseif is_arch("riscv64") then
                local toolchain_file_path = path.join(package:scriptdir(), "cross-cmake","riscv64.toolchain.cmake")
                -- TODO: fixit
                build_cmd = build_cmd.." --rv64"
                build_cmd = build_cmd.." --riscv_toolchain_root /usr/riscv64-linux-gnu"
                build_cmd = build_cmd.." --cmake_extra_defines "..common_cmake_defines.." CMAKE_TOOLCHAIN_FILE="..toolchain_file_path
            else
                build_cmd = build_cmd.." --cmake_extra_defines "..common_cmake_defines
            end
        elseif  package:is_plat("windows") then
                build_cmd = build_cmd.." --cmake_extra_defines "..common_cmake_defines
        end
        os.exec(build_cmd)
        os.exec("cmake --install build/"..build_type)
    end)

package_end()