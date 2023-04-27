(* Basic tests. *)
let () =
  let nn = Network.create [1;6;2] in
  let x = Matrix.of_list [0.1] in
  let o = Matrix.of_list [0.2;0.3] in
  ignore (Network.propagate nn x);
  ignore (Network.backpropagate nn x o);
  ignore (Network.descent nn nn 0.01 x o)

(* Learn f(x) = x². *)
let () =
  let dataset =
    [-1.0; -0.8; -0.6; -0.4; -0.2; 0.0; 0.2; 0.4; 0.6; 0.8; 1.0]
    |> List.map (fun x -> [x], [x *. x])
  in
  let nn = Network.fit
      ~layers:[6;6;6]
      ~precision:1e-5
      ~iterations:10_000
      ~rate:0.2
      dataset
  in
  let tests = [-1.0; 0.6; -0.4; 0.0; 0.1; 0.9; -0.5] in
  List.iter
    (fun x -> Printf.printf "f(%f) = %f vs %f\n" x ((Network.predict nn [x]) |> List.hd) (x *. x))
    tests
