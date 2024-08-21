from .mapper import SearchMapper, SearchMapperInfQuery,SearchMapperWithCrops,SearchMapperWithFixOrgCrops


class PrwMapper(SearchMapper):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)


class PrwSearchMapperInfQuery(SearchMapperInfQuery):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
class PrwCropsMapper(SearchMapperWithCrops):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
        self.SCALES=[[112,48],[192,80],[352,144]]
class PrwFixOrgCropsMapper(SearchMapperWithFixOrgCrops):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
        self.crop_size=(256,128)
