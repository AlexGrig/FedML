class MyMessage(object):
    """
    message type definition
    """

    # server to client
    MSG_TYPE_S2C_SEND_RHS = 1
    MSG_TYPE_S2C_SEND_VAL_REQUEST = 2
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 3
    
    # client to server
    MSG_TYPE_C2S_SEND_TRAIN_PREDICTIONS = 4
    MSG_TYPE_C2S_SEND_TEST_PREDICTIONS = 5

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    #"""
    #    message payload keywords definition
    #"""
    MSG_ARG_KEY_RHS = "righthand_side"
    MSG_ARG_KEY_CLIENT_TRAIN_PREDICTIONS = "client_train_predictions"
    MSG_ARG_KEY_CLIENT_TEST_PREDICTIONS = "client_train_predictions"
