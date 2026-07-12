#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <sstream>
#include <algorithm>
#include <functional>
#include <fstream>

using namespace std;

// Base Abstract Class
class Message
{
public:
    string messageId;
    virtual string get() = 0; // Pure virtual function
    virtual ~Message() {}     // Virtual destructor for safe polymorphic deletion
};

// Derived Class 1: ClientRequest
class ClientRequest : public Message
{
public:
    string clientId;
    string requestTime;

    ClientRequest(string m, string c, string r)
    {
        messageId = m;
        clientId = c;
        requestTime = r;
    }

    string get() override
    {
        stringstream ss;
        // FIX: Replaced string concatenation with stream insertion operators
        ss << messageId << " " << clientId << " " << requestTime;
        return ss.str();
    }
};

// Derived Class 2: ServerResponse
class ServerResponse : public Message
{
public:
    string serverId;
    string metadata;

    ServerResponse(string m, string s, string md)
    {
        messageId = m;
        serverId = s;
        metadata = md;
    }

    string get() override
    {
        stringstream ss;
        ss << messageId << " " << serverId << " " << metadata;
        return ss.str();
    }
};

// Custom Queue Wrapper Class
class CustomQueue
{
private:
    queue<Message *> q;

public:
    void enqueue(Message *m)
    {
        q.push(m);
    }

    void dequeue()
    {
        if (!q.empty())
        {
            Message *frontElement = q.front();
            q.pop();
            delete frontElement; // FIX: Pop first, then delete to prevent undefined state
        }
    }

    string getFront()
    {
        if (!q.empty())
        {
            return q.front()->get();
        }
        return "";
    }

    // Destructor to clean up heap memory safely
    ~CustomQueue()
    {
        while (!q.empty())
        {
            Message *frontElement = q.front();
            q.pop();
            delete frontElement;
        }
    }
};

// --- HackerRank Helper Utility Functions ---
string ltrim(const string &str)
{
    string s(str);
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](unsigned char ch)
                               { return !isspace(ch); }));
    return s;
}

string rtrim(const string &str)
{
    string s(str);
    s.erase(find_if(s.rbegin(), s.rend(), [](unsigned char ch)
                    { return !isspace(ch); })
                .base(),
            s.end());
    return s;
}

vector<string> split(const string &str)
{
    vector<string> tokens;
    string::size_type start = 0;
    string::size_type end = 0;
    while ((end = str.find(" ", start)) != string::npos)
    {
        tokens.push_back(str.substr(start, end - start));
        start = end + 1;
    }
    tokens.push_back(str.substr(start));
    return tokens;
}

// --- Main Processing Loop ---
int main()
{
    string queries_count_temp;
    if (!getline(cin, queries_count_temp))
        return 0;

    int queries_count = stoi(ltrim(rtrim(queries_count_temp)));
    vector<string> queries(queries_count);

    for (int i = 0; i < queries_count; i++)
    {
        string queries_item;
        getline(cin, queries_item);
        queries[i] = queries_item;
    }

    CustomQueue customQueue;
    vector<string> result;

    for (int i = 0; i < queries_count; i++)
    {
        vector<string> query_temp = split(rtrim(queries[i]));
        string type = query_temp[0];

        if (type == "Enqueue")
        {
            string messageType = query_temp[1];

            if (messageType == "C")
            {
                string messageId = query_temp[2];
                string clientId = query_temp[3];
                // Handled string parsing safely without broken stoi conversions
                string requestTime = ltrim(rtrim(query_temp[4]));

                // FIX: Used 'new' keyword to dynamically pass derived class pointer
                customQueue.enqueue(new ClientRequest(messageId, clientId, requestTime));
            }
            else if (messageType == "S")
            {
                string messageId = query_temp[2];
                string serverId = query_temp[3];
                string metadata = query_temp[4];

                // FIX: Used 'new' keyword to dynamically pass derived class pointer
                customQueue.enqueue(new ServerResponse(messageId, serverId, metadata));
            }
        }
        else if (type == "Deque")
        {
            // FIX: Remapped method name from .pop() to .dequeue()
            customQueue.dequeue();
        }
        else if (type == "Top")
        {
            string result_temp = customQueue.getFront();
            result.push_back(result_temp);
        }
    }

    // Output formatting matching HackerRank expectations
    for (size_t i = 0; i < result.size(); i++)
    {
        cout << result[i];
        if (i != result.size() - 1)
        {
            cout << "\n";
        }
    }
    cout << "\n";

    return 0;
}