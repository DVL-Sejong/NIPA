from datetime import date, datetime, timedelta


def get_predict_period_from_dataset(dataset, x_frames, y_frames):
    I_period = dataset['I'].columns.to_list()

    start_date = datetime.strptime(I_period[0], '%Y-%m-%d') + timedelta(days=x_frames)
    end_date = datetime.strptime(I_period[-1], '%Y-%m-%d') + timedelta(days=-y_frames)
    predict_dates = get_period(start_date, end_date, '%Y-%m-%d')
    return predict_dates


def get_period(start_date, end_date, out_date_format=None):
    if type(start_date) == str:
        start_date = datetime.strptime(start_date, get_date_format(start_date))
    if type(start_date) == date:
        start_date = convert_date_to_datetime(start_date)

    if type(end_date) == str:
        end_date = datetime.strptime(end_date, get_date_format(end_date))
    if type(end_date) == date:
        end_date = convert_date_to_datetime(end_date)

    duration = (end_date - start_date).days + 1
    period = [start_date + timedelta(days=i) for i in range(duration)]

    if out_date_format is None:
        return period
    else:
        return [elem.strftime(out_date_format) for elem in period]


def get_date_format(date: str) -> str:
    formats = ['%Y-%m-%d', '%y%m%d', '%m-%d-%Y', '%m/%d/%y']
    for format in formats:
        if validate(date, format):
            return format

    return ''


def validate(date: str, format: str) -> bool:
    try:
        if date != datetime.strptime(date, format).strftime(format):
            raise ValueError
        return True
    except ValueError:
        return False


def convert_date_to_datetime(target_date):
    return datetime(year=target_date.year, month=target_date.month, day=target_date.day)
