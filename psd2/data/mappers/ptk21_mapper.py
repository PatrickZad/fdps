from .mapper import SearchMapper, SearchMapperInfQuery


class Ptk21Mapper(SearchMapper):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)


class Ptk21SearchMapperInfQuery(SearchMapperInfQuery):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
