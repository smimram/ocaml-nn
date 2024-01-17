open Extlib

let () = Random.self_init ()

(* Let's try to learn f(x) = x^2 *)
let () =
  (* Dataset. *)
  let dataset = List.map (fun x -> Vector.scalar x, Vector.scalar (x*.x)) [-1.0; -0.8; -0.6; -0.4; -0.2; 0.0; 0.2; 0.4; 0.6; 0.8; 1.0] in

  (* Train a network with one hidden layer of size 6. *)
  let net = Net.neural ~rate:0.2 [1;6;1] in
  for _ = 0 to 100_000 do
    Net.fit net dataset
  done;

  (* Profit *)
  let xs = [-1.0; -0.5; -0.4; 0.0; 0.1; 0.6; 0.9] in
  List.iter
    (fun x ->
       Printf.printf "f(%f) = %f instead of %f\n" x (Net.predict net (Vector.scalar x) |> Vector.to_scalar) (x *. x)
    ) xs

(*
let () = ignore Net.backward

(* Basic tests. *)
let () =
  Printf.printf "Basic tests...\n%!";
  let nn = Network.create [1;6;2] in
  let x = Matrix.of_list [0.1] in
  let o = Matrix.of_list [0.2;0.3] in
  ignore (Network.propagate nn x);
  ignore (Network.backpropagate nn x o);
  ignore (Network.descent nn nn 0.01 x o)

(* Learn the sine function. *)
let () =
  Printf.printf "Learning sine...\n%!";
  let dataset =
    List.init 100 (fun i -> Float.pi *. float i /. 100.)
    |> List.map (fun x -> [x], [sin x])
    |> List.shuffle
  in
  let nn = Network.fit
      ~layers:[6;6]
      ~precision:1e-4
      ~iterations:10_000
      ~rate:0.1
      dataset
  in
  let tests = List.init 100 (fun i -> Float.pi *. float i /. 100.) in
  List.iter
    (fun x -> Printf.printf "f(%f) = %f vs %f\n" x ((Network.predict nn [x]) |> List.hd) (sin x))
    tests;
  Graphics.open_graph "";
  Graphics.set_color Graphics.green;
  List.iter
    (fun x ->
       let y = sin x in
       let i = int_of_float (x /. Float.pi *. float (Graphics.size_x ())) in
       let j = int_of_float ((y +. 1.) /. 2. *. float (Graphics.size_y ())) in
       Graphics.lineto i j)
    tests;
  Graphics.moveto 0 0;
  Graphics.set_color Graphics.red;
  List.iter
    (fun x ->
       let y = Network.predict nn [x] |> List.hd in
       let i = int_of_float (x /. Float.pi *. float (Graphics.size_x ())) in
       let j = int_of_float ((y +. 1.) /. 2. *. float (Graphics.size_y ())) in
       Graphics.lineto i j)
    tests;
  Graphics.loop_at_exit [Graphics.Key_pressed; Graphics.Button_down] (fun _ -> raise Exit);
*)
