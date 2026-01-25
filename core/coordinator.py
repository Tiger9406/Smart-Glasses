
#coordinator: looks at global queue and processes output from sub workers vision and audio
#kinda like the decision making part

class Coordinator:
    def __init__(self, results_queue):
        self.results_queue = results_queue
        self.past_events = {}
        #self.audio_events
        #self.vision_events
        
        #maybe more; hold past actions taken by self maybe? or a state, like what's happening in the world rn?
        #again, decision making module given the initial processing by the workers

    
    def monitor(self):
        while True:
            if not self.results_queue.empty():
                event = self.results_queue.get()
                self._handle_event(event)

    def _handle_event(self, event):
        #handling events; gotta coordinate event data format
        #for instance if event type is a face in view, we throw it on the picture or sum
        return
    

    