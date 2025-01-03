local header_suffixes = { "h", "hpp" }
local source_suffixes = { "cpp", "cc", "c", "cxx" }
local private_dirs = {"private"}
local binary_dirs = {"bin","lib"}
local binary_suffixes = {"dll","so"}
local test_root_dirs = {"test"}
local test_file_prefixes = {"test_"}

function TargetAddHeaders(base_dir --[[string]])
    for _,private_dir_name in ipairs(private_dirs) do
        local private_dir = path.join(base_dir,private_dir_name)
        add_includedirs(private_dir, { public = false })
        for _,header_suffix in ipairs(header_suffixes) do
            add_headerfiles(path.join(private_dir,"*."..header_suffix), { install = false })
        end
    end
    local public_dir = path.join(base_dir)
    add_includedirs(public_dir, { public = true })
    for _,header_suffix in ipairs(header_suffixes) do
        add_headerfiles(path.join(public_dir,"*."..header_suffix), { install = true })
    end
end


function TargetAddSources(base_dir --[[string]])
    for _,private_dir_name in ipairs(private_dirs) do
        local private_dir = path.join(base_dir,private_dir_name)
        for _,source_suffix in ipairs(source_suffixes) do
            add_files(path.join(private_dir,"*."..source_suffix))
        end
    end
    for _,source_suffix in ipairs(source_suffixes) do
        add_files(path.join(base_dir,"*."..source_suffix))
    end
end


function TargetAddTests(target_name --[[string]],base_dir --[[string]],group_name --[[string]],is_add_deps --[[bool]])
    local test_file_patterns = {}
    for _,test_file_prefix in ipairs(test_file_prefixes) do
        for _,source_suffix in ipairs(source_suffixes) do
            for _,test_root_dir in ipairs(test_root_dirs) do
                table.insert(test_file_patterns,
                path.join(test_root_dir,test_file_prefix.."*."..source_suffix))
            end
        end
    end
    for _,file in ipairs(os.files(table.unpack(test_file_patterns))) do
        local test_target_basename = path.basename(file)
        local test_target_filename = path.filename(file)
        target(test_target_basename)
            set_kind("binary")
            set_default(false)
            if type(group_name) == "string" then               
                set_group(group_name)
            end
            if is_add_deps then
                add_deps(target_name)
            end
            add_files(path.join("test",test_target_filename))
            add_tests("default")
            add_rules("auto_cp_deps_assets_configs_to_build")
            after_load(function(target)
                local testing_target = target:dep(target_name)
                local testing_target_options = testing_target:get("options") or {}
                target:add("options",table.unpack(testing_target_options))
            end)
        target_end()
    end
end


function CreateTarget(target_name --[[string]],kind --[[string]],base_dir --[[string]]
    ,group_name --[[string]],pub_pkgs --[[table]],pub_deps --[[table]],syslinks --[[table]],
    cb_func --[[function]])
    local test_base_dir = path.join(base_dir,"test")
    local test_group_name = group_name.."-tests"
    -- target
    target(target_name)
    set_kind(kind)
    set_group(group_name)
    add_rules("auto_cp_deps_assets_configs_to_build")
    -- packages
    -- add packages here
    if type(pub_pkgs) == "table" then
        for _, pkg_name in ipairs(pub_pkgs) do
            add_packages(pkg_name, { public = true })
        end
    end
    -- add deps here
    if type(pub_deps) == "table" then
        for _, dep_name in ipairs(pub_deps) do
            add_deps(dep_name, { inherit = true })
        end
    end
    TargetAddHeaders(base_dir)
    TargetAddSources(base_dir)
    -- links
    -- add links here
    for _,syslib in ipairs(syslinks) do
        add_syslinks(syslib)
    end
    if type(cb_func) == "function" then
        cb_func()
    end
    target_end()
    
    
    -- tests
    if string.find(kind,"binary") then
        TargetAddTests(target_name,test_base_dir,test_group_name,true)
    else
        TargetAddTests(target_name,test_base_dir,test_group_name,true)
    end
end