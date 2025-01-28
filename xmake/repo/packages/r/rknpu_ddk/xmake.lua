package("rknpu_ddk")
    set_kind("library")
    set_homepage("https://github.com/airockchip/rknpu_ddk")
    set_description("RKNPU DDK is provide advanced interface to access Rockchip NPU.")
    set_license("Apache License 2.0")

    add_configs("shared", { description = "Forced Shared Library", default = true, readonly = true })

    if is_plat("linux") then
        if is_arch("arm64") then
            set_urls("https://github.com/airockchip/rknpu_ddk.git")
        end
    end

    on_install("linux", function(package)
        if is_arch("arm") then
            os.cp("include/*", package:installdir("include"))
            os.cp("lib/*", package:installdir("lib"))
        elseif is_arch("arm64") then
            os.cp("include/*", package:installdir("include"))
            os.cp("lib64/*", package:installdir("lib"))
            local to_lib64_dir = path.join(package:installdir(), "lib64")
            os.mkdir(to_lib64_dir)
            os.cp("lib64/*", to_lib64_dir)
        end
    end)
package_end()