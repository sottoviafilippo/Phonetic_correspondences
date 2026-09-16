# Phonetic_correspondences
A model that learns phonetic correspondences between two languages. 
It does so with a similar (i.e. basically copied) structure to the `Attention is all you need` paper. An excuse for me to learn Transformes, applied to a small dimensional case such as correspondences between text characters. 

- The Italian-Spanish cognate pairs are listed in `it_es_cognates.txt`. A bit more than 2000 cognate words are given for the moment; more would be needed.
- `order_alphabetical` is a Python code that puts the cognate file in alphabetical order and cancel unneeded spaces. 
- The definitions of the classes are in the `encoder_decoder.py` files.
- I test and use them in the `phonetic_correspondences_trials.ipynb` notebook, training `it_es_cognates.txt`. 
One first defines a transformer `transformer = Transformer(n_layers = 4, d_model = 32, n_heads = 8, vocab_size = vocab_size, char_to_idx = char_to_idx, dk = 8, dv = 8, hidden_dim_over_d_model = 2, lr = 0.01, scheduler_step_size=100)`. One then fits it over the data contained in the `.txt` file with `trafo2.fit_from_txt(file_name, n_epoch)`. In the notebook I test and compare models of different sizes and give some short comments about the effect of dropout regularization and weight decay (the latter not seeming to be very useful, for the moment). Dropout seems to help more, but still, overfitting issues. Probably I would need a larger dataset. 



