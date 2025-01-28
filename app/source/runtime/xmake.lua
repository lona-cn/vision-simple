IncludeSubDirs(os.scriptdir())

-- runtime target
target("runtime")
    set_kind("static")
    set_group("runtime")
    add_files("dummy.cpp")
    add_extrafiles(path.join(os.scriptdir(),"xmake.lua"))
    local deps = {"infer","common"}
    for _,dep in ipairs(deps) do 
        add_deps(dep,{inherit = true})
    end
    set_basename("vsrt_$(plat)_$(arch)_$(mode)")
    set_policy("build.merge_archive", true)
target_end()