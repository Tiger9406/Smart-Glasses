# Smart Glasses

Let's lock in

---

## Description: multiprocessing server

This'll be the server to our smart glasses; we stream in raw data via websockets & process it how we want
We ingest and then have parallel workers (visual & audio) preprocessing the incoming data; we use multiprocessing
Then these parallel workers are to produce output events and send them in a queue for another worker: the coordinator, to take action on these events
Exact action split we can discuss later, but I envision probably text-to-speech happens in audio workers and some smaller ml models on visual workers
Later coordinator combine both to make more complex actions

## Setup

Using Python 3.11; other requirements minimal as of rn; reference requirements.txt
I'm gonna use venv instead of conda; feel free to use either though




