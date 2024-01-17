type t = float array

let dim (x:t) = Array.length x

let map f (x:t) : t = Array.map f x

let map2 f (x:t) (y:t) : t = Array.map2 f x y

let add (x:t) (y:t) : t = Array.map2 (+.) x y

let cmul a x = map (fun x -> a *. x) x

let hadamard x y = map2 ( *. ) x y

let init n f : t = Array.init n f
