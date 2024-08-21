from .mapper import SearchMapper, SearchMapperInfQuery,SearchMapperWithCrops,SearchMapperWithFixOrgCrops


class CuhksysuMapper(SearchMapper):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)


class CuhkSearchMapperInfQuery(SearchMapperInfQuery):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)

class CuhksysuCropsMapper(SearchMapperWithCrops):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
        self.SCALES=[[80,48],[176,96],[432,208]]
class CuhksysuFixOrgCropsMapper(SearchMapperWithFixOrgCrops):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
        self.crop_size=(256,128)
