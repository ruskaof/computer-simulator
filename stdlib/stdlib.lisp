(defun print_int (x)
  (while (> x 0)
    (print_char (+ (% x 10) 48))
    (setq x (/ x 10))
  )
)