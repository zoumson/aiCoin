import random



if __name__ == "__main__":
    # print(random.randint(1, 50))
    set_ticket = set()

    while len(set_ticket) < 6:
        tmp = random.randint(1, 50)
        set_ticket.add(tmp)


    list_draw = []

    while len(list_draw) < 7:
        tmp = random.randint(1, 50)
        list_draw.append(tmp)



    count_list = []

    for num in list_draw:
        if num in set_ticket:
            count_list.append(num)


    print(set_ticket)

    print(list_draw)
    print(count_list)
    print(len(count_list))





