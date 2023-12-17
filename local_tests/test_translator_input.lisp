(progn
    (defun foo (x y)
      (progn
        (print_string "foo")
        (print_string x)
        (print_string y)
      )
    )

    (foo "xarg" "yarg")
  )