from pymongo import MongoClient
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
# Пример данных для графа знаний

class MongoVectorDB:
    def __init__(self, db_name, collection_name, host='localhost', port=27017):
        """
        Инициализация подключения к MongoDB и создание базы данных и коллекции.

        :param db_name: Имя базы данных
        :param collection_name: Имя коллекции
        :param host: Адрес сервера MongoDB (по умолчанию 'localhost')
        :param port: Порт сервера MongoDB (по умолчанию 27017)
        """
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_data(self, data):
        """
        Вставка данных в коллекцию.

        :param data: Данные для вставки (dict или list из dict)
        """
        if isinstance(data, list):
            self.collection.insert_many(data)
        else:
            self.collection.insert_one(data)

    def find_data(self, query):
        """
        Поиск данных в коллекции.

        :param query: Запрос для поиска (dict)
        :return: Список найденных документов
        """
        return list(self.collection.find(query))

    def update_data(self, query, new_values):
        """
        Обновление данных в коллекции.

        :param query: Запрос для поиска обновляемых документов (dict)
        :param new_values: Новые значения для обновления (dict)
        """
        self.collection.update_many(query, {'$set': new_values})

    def delete_data(self, query):
        """
        Удаление данных из коллекции.

        :param query: Запрос для удаления документов (dict)
        """
        self.collection.delete_many(query)

    def close_connection(self):
        """
        Закрытие подключения к MongoDB.
        """
        self.client.close()