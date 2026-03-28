"""
으뜸 딥러닝 — 08장 06절
세 모델 학습
"""

rnn_model  = SeqModel('RNN')
lstm_model = SeqModel('LSTM')
gru_model  = SeqModel('GRU')

rnn_losses  = train_model(rnn_model,  train_X, train_y)
lstm_losses = train_model(lstm_model, train_X, train_y)
gru_losses  = train_model(gru_model,  train_X, train_y)
