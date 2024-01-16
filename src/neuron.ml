(** Batch references. *)
module Batch = struct
  type t =
    {
      mutable contents : float;
      mutable batch : float list;
    }

  let make x =
    {
      contents = x;
      batch = []
    }

  let get r = r.contents

  let set r x = r.batch <- x::r.batch

  module Operations = struct
    let ref x = make x
    let (!) = get
    let (:=) = set
  end

  let collect r =
    let s, n = List.fold_left (fun (s,n) x -> s+.x, n+1) (0.,0) r.batch in
    r.contents <- s /. float n;
    r.batch <- []
end
