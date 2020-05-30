using Random
using FileIO
using Blink

check(bound) = cell -> all((1, 1) .≤ cell .≤ bound)
neighbors(cell, bound, step=2) =
    filter(check(bound), collect(map(dir -> cell .+ step .* dir,
                                     ((0, 1), (-1, 0), (0, -1), (1, 0)))))
 
function walk(maze::Matrix, nxtcell, visited=Tuple{Int, Int}[])
    push!(visited, nxtcell)
    ns = shuffle(neighbors(nxtcell, size(maze)))
    for n in ns
        if !(n in visited)
            gap = div.((nxtcell .+ n), 2)
            maze[gap...] = false
            walk(maze, n, visited)
        end
    end
    return maze
end

function maze(h, w)
    maze = [(i|j)&1 == 1 for i in 1:2h+1, j in 1:2w+1]
    firstcell = (2rand(1:h), 2rand(1:w))
    println(firstcell)
    return walk(maze, firstcell)
end

pprint(matrix) = for i = 1:size(matrix, 1) println(join(matrix[i, :])) end
function printmaze(maze)
    walls = split("╹ ╸ ┛ ╺ ┗ ━ ┻ ╻ ┃ ┓ ┫ ┏ ┣ ┳ ╋")
    println(walls)
    h, w = size(maze)
    f = cell -> 2 ^ div((3cell[1] + cell[2] + 3), 2)
    wall(i, j) = if !maze[i,j] " " else
        walls[Int(sum(f, filter(x -> maze[x...], neighbors((i, j), (h, w), 1)) .- (i, j)))]
    end
    mazewalls = collect(wall(i, j) for i in 1:2:h, j in 1:w)
    pprint(mazewalls)
end

function make_image(nh, nw; Wd=1, Cd=1)
    iw = (nw+1)*Wd + nw*Cd
    ih = (nh+1)*Wd + nh*Cd
    img = fill(0xff, (ih, iw))
    alpha = trues(ih, iw)
    mz = maze(nh, nw)
    mzh, mzw = size(mz)

    oi, oj, ci, cj = 0, 0, 0, 0
    for j = 1:mzw
        cj = isodd(j) ? Wd : Cd
        for i = 1:mzh
            ci = isodd(i) ? Wd : Cd
            res = mz[i, j]
            alpha[oi+1:oi+ci, oj+1:oj+cj] .= !res
            img[oi+1:oi+ci, oj+1:oj+cj] .= res ? 0x00 : 0xff
            oi = oi + ci
        end
        oi, oj = 0, oj + cj
    end
    return mz, alpha, img
end

function build_maze(;fn="maze.png")
    _, alpha, maze = make_image(4, 4, Wd=10, Cd=100)
    save(fn, maze)
    return alpha
end

