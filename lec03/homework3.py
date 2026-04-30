def words2characters(words):
    """
    This function converts a list of words into a list of characters.

    @param:
    words - a list of words

    @return:
    characters - a list of characters

    Every element of "words" should be converted to a str, then split into
    characters, each of which is separately appended to "characters." For 
    example, if words==['hello', 1.234, True], then characters should be
    ['h', 'e', 'l', 'l', 'o', '1', '.', '2', '3', '4', 'T', 'r', 'u', 'e']
    """
    characters = []
    for word in words:
        for char in str(word):
            characters.append(char)
    return characters


def cancellation(input_list, stop_word):
    """
    This function copies elements one by one from input_list into output_list.
    If one of the elements is equal to the stop_word, then stop the function
    and return what you have so far.
    """
    output_list = []
    for element in input_list:
        if element == stop_word:
            break
        output_list.append(element)
    return output_list


def copy_all_but_skip_word(input_list, skip_word):
    """
    This function should copy elements one by one from input_list into output_list.
    If one of the elements is equal to the skip_word, then you should skip that element,
    but keep checking all of the other elements.
    """
    output_list = []
    for element in input_list:
        if element == skip_word:
            continue
        output_list.append(element)
    return output_list


def my_average(input_list):
    """
    You may assume that `input_list` is a non-empty list, in which every element is a number.  
    Calculate the average value, and return it.
    """
    total = 0
    for num in input_list:
        total += num
    return total / len(input_list)