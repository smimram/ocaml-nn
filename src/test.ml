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
