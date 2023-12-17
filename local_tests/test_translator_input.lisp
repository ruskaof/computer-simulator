(progn
  (defun print_int (x)
    (progn
      (setq num_to_div 1)
      (while (> (/ x num_to_div) 10)
        (setq num_to_div (* num_to_div 10))
      )
      (while (> x 0)
        (progn
          (print_char (+ 48 (/ x num_to_div)))
          (setq x (% x num_to_div))
          (setq num_to_div (/ num_to_div 10))
        )
      )
    )
  )

  (print_int 123)
)