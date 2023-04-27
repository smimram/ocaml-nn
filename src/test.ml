open Nn

(* Learn f(x) = x². *)
let () =
  let dataset =
    [-1.0; -0.8; -0.6; -0.4; -0.2; 0.0; 0.2; 0.4; 0.6; 0.8; 1.0]
    |> List.map (fun x -> [x], [x *. x])
  in
  let nn = Network.fit
      ~layers:[6] 
      ~eps:1e-5
      ~iter:100_000 
      ~rate:0.2
      dataset 
  in
  let tests = [-1.0; 0.6; -0.4; 0.0; 0.1; 0.9; -0.5] in
  List.iter
    (fun x -> Printf.printf "f(%f) = %f vs %f\n" x ((Network.predict nn [x]) |> List.hd) (x *. x))
    tests
