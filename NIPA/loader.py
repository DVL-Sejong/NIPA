from datetime import timedelta


class DataLoader:
    def __init__(self, data_info, dataset):
        self.test_start = data_info.test_start
        self.test_end = data_info.test_end
        self.x_frames = data_info.x_frames
        self.y_frames = data_info.y_frames

        self.regions = dataset['regions']
        self.initiate(dataset['I'])

    def __len__(self):
        return self.len - (self.x_frames + self.y_frames) + 1

    def initiate(self, I_df):
        self.start = (self.test_start + timedelta(days=-self.x_frames)).strftime('%Y-%m-%d')
        self.end = (self.test_end + timedelta(days=self.y_frames-1)).strftime('%Y-%m-%d')
        self.I_df = I_df.loc[self.regions, self.start:self.end]
        self.len = len(self.I_df.columns.to_list())

    def __getitem__(self, idx):
        idx += self.x_frames
        data = self.I_df.iloc[:, idx-self.x_frames:idx+self.y_frames]

        x = data.iloc[:, :self.x_frames]
        y = data.iloc[:, self.x_frames:]
        test_dates = y.columns.to_list()

        return x, y, test_dates
