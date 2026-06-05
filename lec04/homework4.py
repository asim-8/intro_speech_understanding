def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    # If there are no birthdays recorded, return a default state
    if not birthdays:
        return (1, 1), []

    # Sort all birthday dates chronologically
    sorted_dates = sorted(birthdays.keys())

    # 1. Look for the next birthday in the remaining part of the year
    for bday in sorted_dates:
        if bday >= date:
            return bday, birthdays[bday]

    # 2. Wrap around to the next year (the earliest date in the dict)
    next_year_bday = sorted_dates[0]
    return next_year_bday, birthdays[next_year_bday]