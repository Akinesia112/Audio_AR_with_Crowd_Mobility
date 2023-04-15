using UnityEngine;
using System;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class PlayerControllerScript : MonoBehaviour
{
    // 1. Declare Variables
    Thread receiveThread;
    UdpClient client;
    int port;
    public string text; // public variable to store received text

    // 2. Initialize variables
    void Start()
    {
        port = 5000;
        InitUDPThread();
    }

    // 3. Initialize UDP thread
    private void InitUDPThread()
    {
        print("UDP Initialized");

        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    // 4. Receive Data
    private void ReceiveData()
    {
        client = new UdpClient(port);
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Parse("192.168.32.239"), port);
                byte[] data = client.Receive(ref anyIP);
                string receivedText = Encoding.UTF8.GetString(data);
                print(">> " + receivedText);

                // Store the received text in the public variable
                text = receivedText;
            }
            catch (Exception e)
            {
                print(e.ToString());
            }
        }
    }

    // 5. GetText method to return the value of the "text" variable
    public string GetText()
    {
        return text;
    }
}
