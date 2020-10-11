class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = {'state': [],
                       'action': [],
                       'reward': [],
                       'next_state': [],
                       'done': []}

    def size(self):
        return len(self.buffer['state'])

    def store(self, experience_dict):
        if self.size() >= self.max_size:
            for key in self.buffer.keys():
                self.buffer[key].pop(0)
        for key, value in experience_dict.items():
            self.buffer[key].append(value)
