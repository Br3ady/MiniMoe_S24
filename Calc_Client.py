import grpc
import calc_pb2
import calc_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = calc_pb2_grpc.CalculatorStub(channel)
        response = stub.Add(calc_pb2.AddRequest(a=3, b=5))
        print(f"3 + 5 = {response.result}")

if __name__ == '__main__':
    run()
