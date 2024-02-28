import torch
import json

class Logger(object):
    def __init__(self, runs, info=None,filename = None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.training_info = []
        if filename == None:
            self.filename = str(info.dataset + "_"  + info.method + ".log")
        else:
            self.filename = str(filename + ".log")

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, ratio = 1, run=None):
        result_string = ""
        if run is not None:
            result = ratio * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            result_string += f'Run {run + 1:02d}:\n'
            result_string += f'Highest Train: {result[:, 0].max():.2f}\n'
            result_string += f'Highest Valid: {result[:, 1].max():.2f}\n'
            result_string += f'  Final Train: {result[argmax, 0]:.2f}\n'
            result_string += f'   Final Test: {result[argmax, 2]:.2f}\n'
            self.save_results(result_string)
        else:
            ipdb.set_trace()
            result = ratio * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            result_string += 'All runs:\n'
            r = best_result[:, 0]
            result_string += f'Highest Train: {r.mean():.2f} ± {r.std():.2f}\n'
            r = best_result[:, 1]
            result_string += f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}\n'
            r = best_result[:, 2]
            result_string += f'  Final Train: {r.mean():.2f} ± {r.std():.2f}\n'
            r = best_result[:, 3]
            result_string += f'   Final Test: {r.mean():.2f} ± {r.std():.2f}\n'
            self.save_results(result_string)

    def save_results(self, string_text):
        print(string_text)
        with open(self.filename, 'a+') as file:
            file.write(string_text + '\n')
