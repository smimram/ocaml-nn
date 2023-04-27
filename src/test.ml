(* Basic tests. *)
let () =
  let nn = Network.create [1;6;2] in
  let x = Matrix.of_list [0.1] in
  let o = Matrix.of_list [0.2;0.3] in
  ignore (Network.propagate nn x);
  ignore (Network.backpropagate nn x o);
  ignore (Network.descent nn nn 0.01 x o)

(* Learn the sine function. *)
let () =
  let dataset =
    List.init 100 (fun i -> 3. *. float i /. 100.)
    |> List.map (fun x -> [x], [sin x])
  in
  let nn = Network.fit
      ~layers:[6;6]
      ~precision:1e-5
      ~iterations:10_000
      ~rate:0.2
      dataset
  in
  let tests = List.init 10 (fun i -> 3. *. float i /. 10.) in
  List.iter
    (fun x -> Printf.printf "f(%f) = %f vs %f\n" x ((Network.predict nn [x]) |> List.hd) (sin x))
    tests
