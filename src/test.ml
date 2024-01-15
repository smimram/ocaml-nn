module List = struct
  include List

  (** Shuffle the elements of a list. *)
  let shuffle l =
    l
    |> List.map (fun x -> Random.bits (), x)
    |> List.sort Stdlib.compare
    |> List.map snd
end

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
