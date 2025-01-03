package("onnxruntime-dml")
-- https://www.nuget.org/packages/Microsoft.AI.DirectML/
set_kind("library")
set_homepage("https://www.onnxruntime.ai")
set_description("ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator")
set_license("MIT")

add_configs("dml", { description = "Enable MicroSoft DirectML Support", default = true, readonly = true })
add_configs("shared", { description = "Forced Shared Library", default = true, readonly = true })
add_configs("runtimes", { description = "Forced MD", default = "MD", readonly = true })
add_deps("directml",{system = false})
if is_plat("windows") then
    if is_arch("x64") then
        set_urls("https://github.com/microsoft/onnxruntime/releases/download/v$(version)/Microsoft.ML.OnnxRuntime.DirectML.$(version).nupkg")
        add_versions("1.20.0", "be368111c64495330d0ff9ad11857b37c406cb143204b0a65be418ac86e99e93")
    end
end

on_download(function(package, opt)
    import("net.http")
    import("utils.archive")
    local url = opt.url
    local sourcedir = opt.sourcedir
    local packagefile = path.filename(url)
    packagefile = string.gsub(packagefile, "nupkg", "zip")
    local sourcehash = package:sourcehash(opt.url_alias)

    local cached = true
    if not os.isfile(packagefile) or sourcehash ~= hash.sha256(packagefile) then
        cached = false

        -- attempt to remove package file first
        os.tryrm(packagefile)
        http.download(url, packagefile)

        -- check hash
        if sourcehash and sourcehash ~= hash.sha256(packagefile) then
            raise("unmatched checksum, current hash(%s) != original hash(%s)", hash.sha256(packagefile):sub(1, 8), sourcehash:sub(1, 8))
        end
    end

    -- extract package file
    local sourcedir_tmp = sourcedir .. ".tmp"
    os.rm(sourcedir_tmp)
    if archive.extract(packagefile, sourcedir_tmp) then
        os.rm(sourcedir)
        os.mv(sourcedir_tmp, sourcedir)
    else
        -- if it is not archive file, we need only create empty source file and use package:originfile()
        os.tryrm(sourcedir)
        os.mkdir(sourcedir)
    end

    -- save original file path
    package:originfile_set(path.absolute(packagefile))
end)

on_install("windows", function(package)
    os.cp("runtimes/win-x64/native/*.lib", package:installdir("lib"))
    os.cp("runtimes/win-x64/native/*.dll", package:installdir("bin"))
    os.cp("build/native/include/*", package:installdir("include"))
end)

on_test(function(package)
    assert(package:check_cxxsnippets({ test = [[
            #include <array>
            #include <cstdint>
            void test() {
                std::array<float, 2> data = {0.0f, 0.0f};
                std::array<int64_t, 1> shape{2};

                Ort::Env env;

                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                auto tensor = Ort::Value::CreateTensor<float>(memory_info, data.data(), data.size(), shape.data(), shape.size());
            }
        ]] }, { configs = { languages = "c++17" }, includes = "onnxruntime_cxx_api.h" }))
end)
package_end()