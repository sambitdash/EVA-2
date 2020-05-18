using FileIO
using Blink
using LinearAlgebra
using BSON: @load, @save
#using Zygote
#using CUDAdrv
#using CuArrays

include("DQN.jl")

struct Car
    img::Matrix
    sensors::NTuple{6, Tuple{Float32, Float32}}
    function Car()
        img = load("car.png")
        l, w = size(img)
        sd = 10f0
        sensors = ((-w/2-sd,  -sd), (0,   -sd), (w/2+sd,  -sd),
                   (-w/2-sd, l+sd), (0,  l+sd), (w/2+sd, l+sd))
        return new(img, sensors)
    end
end

struct CarState
    h::Float32
    w::Float32
    loc::Tuple{Float32, Float32}
    goal::Tuple{Float32, Float32}
    ang::Float32 #Clockwise positive
end

function CarState(c::Car, loc=(0f0, 0f0), goal=(410f0, 410f0), ang=0f0)
    h, w = size(c.img)
    return CarState(h, w, loc, goal, ang)    
end

function dist2(s::CarState)
    loc = get_curr_loc(s, (0, s.h)) 
    v = s.goal .- loc
    return dot(v, v)
end

ANG_STEP = 10f0

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

left(s::CarState)  = rotate(s, -pi/180f0*ANG_STEP)
right(s::CarState) = rotate(s,  pi/180f0*ANG_STEP)

forward(s::CarState, d) = CarState(s.h, s.w, s.loc .+ (-d*sin(s.ang), d*cos(s.ang)),
                                   s.goal, s.ang)

function get_curr_loc(s::CarState, v)
    sth, cth, ox, oy = sin(s.ang), cos(s.ang), s.loc[1], s.loc[2]
    return v[1]*cth - v[2]*sth + ox, v[1]*sth + v[2]*cth + oy
end

get_curr_sensors(c::Car, s::CarState) = get_curr_loc.(Ref(s), c.sensors)

FWD_SPEED = 200f0                        # Pixels per sec
BK_SPEED_FACTOR = 10f0                   # Ratio of forward and backward speed
BK_SPEED  = FWD_SPEED / BK_SPEED_FACTOR  
ACTION_EVAL_INTERVAL = 10f0/FWD_SPEED     # Average check every 10 fwd pixels
ACTIONS = (:forward, :left, :right, :back)

roadmap = Matrix{Bool}(load("maze.png"))

l_m, w_m = size(roadmap)
println(l_m, " ", w_m)

w = Window()
loadfile(w, joinpath(@__DIR__, "file.html"))

goal = (400f0, 400f0)
bloc = (50f0, 50f0)

episode = 0

car = Car()
cs  = CarState(car, bloc, goal, 0f0)

dqn = DQN{10, length(ACTIONS)}(0.99)

#if isfile("model-weights.bson")
#    @load "model-weights.bson" weights
#    Flux.loadparams!(dqn.model, weights)
#end

last_reward = 0f0
payout = 0f0
ntime = 0
episodes = Vector{Tuple{Int, Int, Float32, Float32}}()
episode_state = :e

inited = false
tm = time()

valid_sensor(sensor) =
    (global l_m, w_m; 1 <= sensor[1] <= w_m && 1 <= sensor[2] <= l_m)

sensor_on_wall(sensor) = (global roadmap;
                          !roadmap[round(Int, sensor[2]), round(Int, sensor[1])])

println(:none, " ", cs)

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

function reward(ps::CarState, s::CarState, a)
    global car
    sensors = get_curr_sensors(car, s)
    all(valid_sensor, sensors) || return (-1f0, true)
    car_on_wall(car, s) && return (-0.9f0, false)
    #any(sensor_on_wall, sensors) && return (-1f-2, false)
    d, dp = dist2(s), dist2(ps)
    d < 625f0 && return (1f0, true)
    d < dp && return (25f-3, false)
    return (-25f-3, false)
end

function signal(s::CarState)
    global car, l_m, w_m
    sensors = get_curr_sensors(car, s)
    loc = get_curr_loc(s, (0, s.h)) 
    v = s.goal .- loc
    v1 = map(sensors) do sensor
        y, x = round.(Int, sensor)
        return !(12 <= x <= w_m-12 && 12 <= y <= l_m-12) ? 1f0 :
            count(roadmap[x-5:x+5, y-5:y+5] .== false)/121f0
    end
    println(v, " ", v1)
    return v..., s.ang, v1..., car_on_wall(car, s) ? 1f0 : 0f0
end

function eval_action()
    global cs, car, w, bloc, goal, tm, episode, last_reward, dqn, payout, ntime
    global episode_state

    while !dqn.trained
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
        action = ACTIONS[update!(dqn, last_reward, signal(cs), episode_state)]
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

        payout = dqn.gamma*payout + r

        println("episode = $episode ", action, " ", ns, " ", d, " nmem:", dqn.memory.n,
                " reward:", r, " payout: ", payout)

        if isterminal
            episode_state = :e
            act = signal(ns)
            update!(dqn, r, act, episode_state)
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


