#!/usr/bin/env python3
from qunetsim.components import Host
from qunetsim.components import Network
from qunetsim.objects import Qubit
from qunetsim.objects import Logger

# Set to False, to get more information
Logger.DISABLED = True


def sender_protocol(host, receiver):
    secret = "It must be remembered that there is nothing more difficult to plan, more doubtful of success, nor more dangerous to manage, than the creation of a new system. For the initiator has the enmity of all who would profit by the preservation of the old institutions, and merely lukewarm defenders in those who would gain by the new ones."
    # secret = "test"
    
    secret_bin = list(map(bin, bytearray(secret, 'utf-8')))
    secret_bin = [x[2:].zfill(8) for x in secret_bin]

    # Sending the secret
    for character in secret_bin:
        print(f"{host.host_id}: sending a character: {character}")
        for bit in character:
            # TODO: Create a qubit and encode the classical bit into it.
            # Note: a qubit is created in the stata |0> by default
            q = Qubit(host)
            if (bit=="1"):
                q.X()
            # TODO: Send the qubit to the receiver, make it await acknowledgment
            # Send the qubit and await an ACK
            host.send_qubit(receiver, q, await_ack=True)



    # TODO: Send the classical message to the receiver
    # THe content of the message should be "END"
    # The message signals the end of secret phrase transmission
    print(f"Sending classical END")
    host.send_classical(receiver, 'END')

    # Secret Verify
    # TODO: Receive classical message, which includes the secret
    message = host.get_next_classical(receiver)

    recv_secret = message.content
    print()
    if recv_secret==secret:
        print(f"{host.host_id}: Secret Exchange succeeded")
        print(f"Secret: {secret}")
    else:
        print(f"{host.host_id}: Secret Exchange Failed")
        print(f"Secret sent; {secret}")
        print(f"Secret received: '{recv_secret}'")


def receiver_protocol(host, sender):
    secret_bits = []
    while True:
        classical_message = host.get_classical(sender, wait=0)
        if len(classical_message) > 0:
            break
        # TODO: Get the qubit which was sent by the sender
        # Use the get_data_qubit(sender, wait) method
        # Set wait parameter to 5
        q = host.get_data_qubit(sender, wait=5)

        if q is None:
            continue
        # TODO: Measure the qubit and append it to the secret_bits list
        m = q.measure()
        secret_bits.append(m)

        if len(secret_bits)%8 == 0:
           sb = ["".join(map(str, secret_bits[i:i+8])) for i in range(0,len(secret_bits), 8)]
           print(f"{host.host_id}: received a character: {sb[-1]}")


    # Decoding the secret
    secret_bits = ["".join(map(str, secret_bits[i:i+8])) for i in range(0,len(secret_bits), 8)]
    print(f"{host.host_id} received the following bits:")
    print("\n".join(" ".join(secret_bits[i:i+6]) for i in range(0,len(secret_bits), 6)))
    secret_chars = [chr(int(s,2)) for s in secret_bits]
    secret = "".join(secret_chars)

    # Secret Verify
    # TODO: Send the secret (variable secret) back to the sender for verification.
    print("Secret from receiver: ", secret)
    host.send_classical(sender, secret)


def main():
    # TODO: get the Network() instance
    network = Network.get_instance()
    network.start()


    # TODO: Choose the names of the two nodes
    # write them in the nodes list as strings
    nodes = ['Alice', 'Bob']

    # TODO: start the network with the list of nodes
    network.start(nodes)

    # TODO: Configure the hosts:
    # 1. Create host instances,
    alice = Host('Alice')
    bob = Host('Bob')
    # 2. Create connections between hosts,
    alice.add_connection(bob.host_id)
    bob.add_connection(alice.host_id)
    # 3. Start all of the hosts instances,
    alice.start()
    bob.start()
    # 4. Add hosts to the network.
    network.add_hosts([alice, bob])

    # Configuring First Host

    # TODO: Apply the protocols
    # Each host instance has a run_protocol method, which takes protocol
    # function as first parameter and arguments as second parameter.
    # The first parameter of the protocol should be the host. Self reference
    # is passed automatically.
    # 1. Apply sender protocol to first host,
    # 2. Apply receiver protocol to the second host.
    p1 = alice.run_protocol(sender_protocol, (bob.host_id, ))
    p2 = bob.run_protocol(receiver_protocol, (alice.host_id, ))

    # run_protocol() method returns a thread object. Store both threads as some variable
    # and join them.
    
    p1.join()
    p2.join()

    # start_time = time.time()
    # while time.time() - start_time < 150:
    #     pass


    # TODO: Finally stop the network
    network.stop()


if __name__ == "__main__":
    # Script entry point
    main()
