using FileIO
using Blink
using LinearAlgebra
using OffsetArrays: no_offset_view
using BSON: @load, @save
using Images

#include("td3.jl")

struct Car
    img::Matrix
    gray::Matrix
    function Car()
        img = load("car.png")
        gray = Gray.(img)
        return new(img, gray)
    end
end

struct CarState
    h::Float32
    w::Float32
    loc::Tuple{Float32, Float32}
    goal::Tuple{Float32, Float32}
    ang::Float32 #Clockwise positive
end

function CarState(c::Car, loc=(0f0, 0f0), goal=(400f0, 400f0), ang=0f0)
    h, w = size(c.img)
    return CarState(h, w, loc, goal, ang)    
end

function dist2(s::CarState)
    loc = get_curr_loc(s, (0, s.h)) 
    v = s.goal .- loc
    return dot(v, v)
end

function rotate(s::CarState, angle)
    ang = s.ang + angle
    while ang < -pi
        ang += 2pi
    end
    while ang > pi
        ang -= 2pi
    end
    return CarState(s.h, s.w, s.loc, s.goal, ang)
end

function get_curr_loc(s::CarState, v)
    sth, cth, ox, oy = sin(s.ang), cos(s.ang), s.loc[1], s.loc[2]
    return v[1]*cth - v[2]*sth + ox, v[1]*sth + v[2]*cth + oy
end
#=
FWD_SPEED = 200f0                        # Pixels per sec
BK_SPEED_FACTOR = 10f0                   # Ratio of forward and backward speed
BK_SPEED  = FWD_SPEED / BK_SPEED_FACTOR  
ACTION_EVAL_INTERVAL = 10f0/FWD_SPEED     # Average check every 10 fwd pixels
ACTIONS = (:forward, :left, :right, :back)

roadmap = load("maze.png")

l_m, w_m = size(roadmap)
println(l_m, " ", w_m)

#w = Window()
#loadfile(w, joinpath(@__DIR__, "file.html"))

goal = (400f0, 400f0)
bloc = (50f0, 50f0)

episode = 0

car = Car()
cs  = CarState(car, bloc, goal, 0f0)

last_reward = 0f0
payout = 0f0
ntime = 0
episodes = Vector{Tuple{Int, Int, Float32, Float32}}()
episode_state = :e

inited = false
tm = time()

println(:none, " ", cs)
=#

function plotLine(f::Function, p1, p2)
    x0, y0 = p1
    x1, y1 = p2
    dx =  abs(x1 - x0)
    sx = x0 < x1 ? 1 : -1
    dy = -abs(y1 - y0)
    sy = y0 < y1 ? 1 : -1
    err = dx + dy  # error value e_xy 
    while true
        f(x0, y0) && return true
        x0 == x1 && y0 == y1 && break
        e2 = 2*err;
        if e2 >= dy 
            err += dy # e_xy+e_x > 0 
            x0 += sx
        end
        if e2 <= dx  # e_xy+e_y < 0
            err += dx
            y0 += sy
        end
    end
    return false
end

function car_on_wall(c::Car, s::CarState)
    h, w = size(c.img)
    w2 = div(w, 2)
    p = map([(-w2, 0), (w2, 0), (w2, h), (-w2, h)]) do v
        loc = get_curr_loc(s, v)
        return round(Int, loc[1]), round(Int, loc[2])
    end

    f = (x, y)->!roadmap[y, x]
    
    return (plotLine(f, p[1], p[2]) || plotLine(f, p[2], p[3]) ||
            plotLine(f, p[3], p[4]) || plotLine(f, p[4], p[1])) 
end

function reward(ps::CarState, s::CarState, car::Car, a)
    sensors = get_curr_sensors(car, s)
    car_on_wall(car, s) && return (-0.9f0, false)
    #any(sensor_on_wall, sensors) && return (-1f-2, false)
    d, dp = dist2(s), dist2(ps)
    d < 625f0 && return (1f0, true)
    d < dp && return (25f-3, false)
    return (-25f-3, false)
end

function signal(s::CarState)
    roadmap = load("maze.png")
    l_m, w_m = size(roadmap)

    h = s.h
    loc = get_curr_loc(s, (0, h)) 
    v = s.goal .- loc

    bkimg = zeros(Gray{N0f8}, (80, 80))
    
    loc1 = get_curr_loc(s, (0, div(h, 2)))

    midi = round.(Int, loc1)

    minx, miny, maxx, maxy = midi[1]-40, midi[2]-40, midi[1]+39, midi[2]+39

    cminx, cminy, cmaxx, cmaxy =
        max(minx, 1), max(miny, 1), min(maxx, w_m), min(maxy, l_m)

    rdimg = @view roadmap[cminy:cmaxy, cminx:cmaxx]

    bkimg[cminy-miny+1:cmaxy-miny+1, cminx-minx+1:cmaxx-minx+1] = rdimg
    
    ang = s.ang
    ang < 0 && (ang += 2*pi)
    carimg = no_offset_view(imrotate(Car().gray, ang, 0.75f0))

    h, w = size(carimg)

    oy, ox = 40-div(h, 2), 40-div(w, 2)

    for i=1:h
        for j=1:w
            if abs(carimg[i, j] - 0.75f0) > 0.01f0
                bkimg[oy+i, ox+j] = carimg[i, j]
            end
        end
    end
    return bkimg
end

#=
function eval_action()
    global cs, car, w, bloc, goal, tm, episode, last_reward, payout, ntime
    global episode_state

    while !trained
        if episode_state === :e
            episode += 1
            ntime = 1
            cs = CarState(car, bloc, goal, 0f0)
            last_reward = 0f0
            payout = 0f0
            @js w carview.drawp(0, $(bloc...), true)
            episode_state = :b
            continue
        end

        ntime += 1
        action = ACTIONS[rand(1:4)]#ACTIONS[update!(dqn, last_reward, signal(cs), episode_state)]
        println(action)
        episode_state = :c
        speed = action == :back ? -BK_SPEED : FWD_SPEED
        tm1 = time()
        dtm = (tm1 - tm)
        println(dtm)
        dtm > 100f-3 && (dtm = 100f-3)
        d = speed*dtm
        tm = tm1
        println("distance: ", d)
        ns = cs

        action == :left  && (ns = left(ns))
        action == :right && (ns = right(ns))
        if action == :forward || action == :back
            ns = forward(ns, d)
        end

        r, isterminal = reward(cs, ns, action)

        payout = gamma*payout + r

        #println("episode = $episode ", action, " ", ns, " ", d, " nmem:", dqn.memory.n,
        #        " reward:", r, " payout: ", payout)

        if isterminal
            episode_state = :e
            act = signal(ns)
            #update!(dqn, r, act, episode_state)
            push!(episodes, (episode, ntime, payout, r))
            ntime = 0
            yield()
            continue
        end

        last_reward = r
        cs = ns
        ang = action == :left  ? -ANG_STEP :
              action == :right ?  ANG_STEP : 0
        yd  = (action == :forward || action == :back) ? d : 0
        println(ang, " ",  yd)
        @js w carview.drawp($ang, 0, $yd)
    end
end

#=
tsk_eval  = @task eval_action()
tsk_wait  = @task begin
    v = read(stdin, 1)
    dqn.trained = true
    weights = params(dqn.model);
    @save "model-weights.bson" weights
end

schedule(tsk_eval)
schedule(tsk_wait)

wait(tsk_wait)
=#

#=
=#
=#

using Flux

function test()
    model = Chain(
        # 80x80 image
        Conv((3, 3),  1=>16, pad=(1, 1), relu),
        #BatchNorm(16, relu),
        MaxPool((2,2)),
        # 40x40 image
        Conv((3, 3), 16=>32, pad=(1, 1), relu),
        #BatchNorm(32, relu),
        MaxPool((2,2)),
        # 20x20 image
        Conv((3, 3), 32=>32, pad=(1, 1), relu),
        #BatchNorm(32, relu),
        MaxPool((2,2)),
        # 10x10 image
        Conv((3, 3), 32=>32, pad=(1, 1), relu),
        #BatchNorm(32, relu),
        MaxPool((2,2)),
        # 5x5 image
        Conv((3, 3), 32=>1),
        MeanPool((3, 3)),
        flatten,
        x->pi*tanh.(x)
    )

    opt = ADAM()

    cnt = 1

    pms = params(model)
    trainX = Array{Float32}(undef, (80, 80, 1, 128))
    trainY = Array{Float32}(undef, (1, 128))
    car = Car()

    for j=1:300
        println("batch: ", j)
        
        for i=1:128
            th = (rand() - 5f-1)*2pi
            cs  = CarState(car, (50, 50), (400, 400), th)
            img = signal(cs)
            trainX[:, :, :, i] = img
            trainY[1, i] = th
        end
        
        # println(size(trainX))
        gs = gradient(pms) do
            y = model(trainX)
            v1 = Flux.mse(y, trainY)
            v2 = 1f-3*dot(pms, pms)
            println("mse: ", v1, " ", v2, " ", v1+v2)
            return v1 + v2
        end
        Flux.update!(opt, pms, gs)
    end

    for i=1:16
        th = (rand() - 5f-1)*2pi
        cs  = CarState(car, (50, 50), (400, 400), th)
        img = signal(cs)
        trainX[:, :, :, i] = img
        trainY[1, i] = th
    end

    y = model(trainX)

    for (a, b) in zip(y, trainY)
        println(a, " ", b)
    end
end
