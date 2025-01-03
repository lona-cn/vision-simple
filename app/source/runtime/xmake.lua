IncludeSubDirs(os.scriptdir())

-- runtime target
target("runtime")
set_kind("static")
set_group("runtime")
local deps = {"common","config","infer","server"}
for _,dep in ipairs(deps) do 
    add_deps(dep,{inherit = true})
end
set_policy("build.merge_archive", true)
target_end()