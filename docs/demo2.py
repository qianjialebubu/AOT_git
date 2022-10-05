import torch
import os
torch.save(self.netG.module.state_dict(),
    os.path.join(self.args.save_dir, f'G{str(self.iteration).zfill(7)}.pt'))
torch.save(self.netD.module.state_dict(),
    os.path.join(self.args.save_dir, f'D{str(self.iteration).zfill(7)}.pt'))
torch.save(
    {'optimG': self.optimG.state_dict(), 'optimD': self.optimD.state_dict()},
    os.path.join(self.args.save_dir, f'O{str(self.iteration).zfill(7)}.pt'))