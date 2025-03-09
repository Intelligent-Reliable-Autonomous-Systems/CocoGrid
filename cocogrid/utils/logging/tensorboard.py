
class MockSummaryWriter:
    def add_scalar(self, label, scalar, global_step, *args, **kwargs):
        print(f'[{global_step}] Logged {label}={scalar}')

    def add_figure(self, label, figure, global_step, close, *args, **kwargs):
        print(f'[{global_step}] Logged {label}=Figure {figure}, close={close}')