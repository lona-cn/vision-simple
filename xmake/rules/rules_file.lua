
rule("auto_cp_deps_assets_configs_to_build")
after_build(function(target)
    local binary_dirs = {"bin","lib"}
    local binary_suffixes = {"dll","so"}
    local function TargetCopyPkgBinariesToBuild(target --[[table]])
        local target_pkgs = target:get("packages")
        local target_dir = target:targetdir()
        local to_dir = target_dir.."/"
        for _,pkg_name in ipairs(target_pkgs) do
            local pkg = target:pkg(pkg_name)
            local install_dir = pkg:installdir()
            if install_dir then
                for _,binary_dir in ipairs(binary_dirs) do
                    for _,binary_suffix in ipairs(binary_suffixes) do
                        os.trycp(path.join(install_dir,binary_dir,"*."..binary_suffix),to_dir)
                    end
                end
            end
        end
    end

    local function TargetCopyAssetsToBuild(target --[[table]])
        local target_dir = target:targetdir()
        local to_dir = target_dir.."/assets/"
        local group = target:get("group")
        local asset_dir = path.join(os.projectdir(),"app","assets")
        os.trycp(asset_dir.."/main/**",to_dir)
        -- add test assests
        if type(group) == "string" and string.find(group,"test") then
            os.trycp(asset_dir.."/test/**",to_dir)
        end
    end

    local function TargetCopyConfigsToBuild(target --[[table]])
        import("core.project.config")
        local target_dir = target:targetdir()
        local target_plat = target:get("plat") or config.plat()
        local target_arch = target:get("arch") or config.arch()
        local to_dir = target_dir.."/config/"
        local config_root_dir = path.join(os.projectdir(),"app","config")
        local config_dirs = {path.join(config_root_dir,"base")}
        local config_plat_dir = path.join(config_root_dir,target_plat.."_"..target_arch)
        if os.exists(config_plat_dir) then
            table.insert(config_dirs,config_plat_dir)
        end
        for _,config_dir in ipairs(config_dirs) do
            os.trycp(config_dir.."/**",to_dir)
        end
    end

    TargetCopyPkgBinariesToBuild(target)
    TargetCopyAssetsToBuild(target)
    TargetCopyConfigsToBuild(target)
end)
rule_end()