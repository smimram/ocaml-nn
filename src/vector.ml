type t = float array

let dim (x:t) = Array.length x

let map f x = Array.map f x

let add (x:t) (y:t) : t = Array.map2 (+.) x y

let init n f : t = Array.init n f
