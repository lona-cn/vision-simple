package("directml")
-- https://www.nuget.org/packages/Microsoft.AI.DirectML/
set_kind("library")
set_homepage("https://learn.microsoft.com/en-us/windows/ai/directml/dml")
set_description("Direct Machine Learning (DirectML) is a low-level API for machine learning (ML)")
-- set_license("MIT")

add_configs("shared", { description = "Forced Shared Library", default = true, readonly = true })
add_configs("runtimes", { description = "Forced MD", default = "MT", readonly = true })

if is_plat("windows") then
    if is_arch("x64") then
        set_urls("https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/$(version)")
        add_versions("1.15.4", "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9")
    end
end

on_download(function(package, opt)
    import("net.http")
    import("utils.archive")
    local url = opt.url
    local sourcedir = opt.sourcedir
    local packagefile = path.filename(url)
	packagefile = "microsoft.ai.directml."..packagefile..".zip"
	print("packagefile: "..packagefile)
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
    -- local sourcedir_tmp = sourcedir .. ".tmp"
    -- os.rm(sourcedir_tmp)
    os.tryrm(sourcedir)
    if archive.extract(packagefile, sourcedir) then
        -- os.rm(sourcedir)
        -- os.mv(sourcedir_tmp, sourcedir)
    else
        -- if it is not archive file, we need only create empty source file and use package:originfile()
        os.tryrm(sourcedir)
        os.mkdir(sourcedir)
    end

    -- save original file path
    package:originfile_set(path.absolute(packagefile))
end)

on_install("windows", function(package)
    if package:is_plat("windows") then
        os.cp("bin/x64-win/*.lib", package:installdir("lib"))
		os.cp("bin/x64-win/*.pdb", package:installdir("lib"))
		os.cp("bin/x64-win/*.dll", package:installdir("bin"))
    end
    os.cp("include/*", package:installdir("include"))
end)

package_end()