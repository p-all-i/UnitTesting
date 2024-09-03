import redis, datetime, json, os

# Defining the fucntion for communication varinet change using pubsub
def subscribe_varientchange(main, loggerObj, host):
    r = redis.StrictRedis(host=host)
    print(f"[INFO] {datetime.datetime.now()} Starting redis pubsub Thread")
    loggerObj.logger.info(f"Starting redis pubsub Thread")
    pubsub = r.pubsub()
    pubsub.subscribe(os.getenv("PUBSUB_CHANNEL"))
    print(f"[INFO] {datetime.datetime.now()} Started redis pubsub Thread")
    loggerObj.logger.info(f"Started redis pubsub Thread")
    print(f"[INFO] {datetime.datetime.now()} redis pubsub Thread listening on transmitter channel")
    loggerObj.logger.info(f"redis pubsub Thread listening on transmitter channel")

    for message in pubsub.listen():
        if message['type'] == 'message':
                print(f"[INFO] {datetime.datetime.now()} Received data for Varient change")
                loggerObj.logger.info(f"Received data for Varient change")
                message = json.loads(message['data'].decode('utf-8'))
                print(message)
                main.varient_change = True
                main.Varient_change_data = message
