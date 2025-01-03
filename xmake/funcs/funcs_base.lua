
function IncludeSubDirs(base_dir --[[string]])
    for _,dir in ipairs(os.dirs(base_dir.."/*")) do
        includes(path.basename(dir))
    end
end