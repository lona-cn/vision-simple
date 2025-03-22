function IncludeSubDirs(base_dir --[[string]])
    for _,dir in ipairs(os.dirs(base_dir.."/*")) do
        if os.exists(path.join(dir,"xmake.lua")) then
            includes(path.basename(dir))
        end
    end
end