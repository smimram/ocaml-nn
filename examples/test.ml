open Nn

let () =
  (* Test softmax, see
     https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/ *)
  let l = Vector.of_list [1.; 2.; 3.] in
  Printf.printf "%s -> %s\n%!" (Vector.to_string l) (Vector.to_string (Vector.softmax l));
  let l = Vector.of_list [1000.; 2000.; 3000.] in
  Printf.printf "%s -> %s\n%!" (Vector.to_string l) (Vector.to_string (Vector.softmax l))
